use std::collections::HashMap;
pub use wasm_bindgen_rayon::init_thread_pool;
use rayon::prelude::*;

use glam::{Vec2, Vec3};
use wasm_bindgen::{prelude::*, JsCast};
use winit::event::{
    ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::EventLoop;
use winit::window::Window;

pub use numcodecs_wasm::*;
pub use zarr_wasm::zarr::{DimensionArraySelection, GetOptions, ZarrArray};

pub mod event;
pub mod renderer;
pub mod util;

use crate::event::{Event, RawArrayReceived};
use util::init;
use util::window;

use crate::event::handler::register_default_js_event_handlers;
use crate::renderer::camera::{Camera, CameraView, Projection};
use crate::renderer::geometry::Bounds3D;
use crate::renderer::volume::RawVolumeBlock;
use crate::renderer::Renderer;
use crate::renderer::wgsl::WGSLPreprocessor;
use crate::window::window_builder_without_size;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// This function is run when the WASM module is instantiated.
#[wasm_bindgen(start)]
pub fn initialize() {
    // note: init stuff used to happen here, but when using WebAssembly threads this function is
    // called once per thread and not once globally, so this moved to `main
}

/// The `GLOBAL_EVENT_LOOP_PROXY` is a means to send data to the running application.
/// It is initialized by `start_event_loop`.
static mut GLOBAL_EVENT_LOOP_PROXY: Option<winit::event_loop::EventLoopProxy<Event<()>>> = None;

///
#[wasm_bindgen(js_name = "dispatchChunkReceived")]
pub fn dispatch_chunk_received(data: Vec<u16>, shape: Vec<u32>) {
    unsafe {
        if let Some(event_loop_proxy) = GLOBAL_EVENT_LOOP_PROXY.as_ref() {
            event_loop_proxy
                .send_event(Event::RawArray(RawArrayReceived { data, shape }))
                .ok();
        } else {
            log::error!("dispatchChunkReceived called on uninitialized event loop");
        }
    }
}

// todo: lib.rs should contain only init and event_loop stuff
//  communication with JS side should only be done via events
//  create CustomEvent enum that consists of WindowEvent and can be extended by other events
//  create thread pool that is supplied with event proxies sending events to event loop
#[wasm_bindgen]
pub fn main(canvas: JsValue) {
    // todo: make logger configurable
    init::set_panic_hook();
    init::set_logger(None);

    log::info!("shader {}",
        WGSLPreprocessor::default()
        .include("type_alias.wgsl", include_str!("renderer/wgsl/type_alias.wgsl"))
        .include("foo.wgsl", include_str!("renderer/wgsl/foo.wgsl"))
        .include("bar.wgsl", include_str!("renderer/wgsl/bar.wgsl"))
        .include("baz.wgsl", include_str!("renderer/wgsl/baz.wgsl"))
        .preprocess("@include(baz.wgsl)").err().unwrap()
    );

    wasm_bindgen_futures::spawn_local(start_event_loop(canvas));
}

async fn start_event_loop(canvas: JsValue) {
    let zarr_array = ZarrArray::open(
        "http://localhost:8005/".to_string(),
        "ome-zarr/m.ome.zarr/0/2".to_string(),
    )
    .await;
    log::info!("ZarrArray {:?}", zarr_array.shape());

    let selection = vec![
        DimensionArraySelection::Number(0.),
        DimensionArraySelection::Number(0.),
    ];
    let raw = zarr_array
        .get_raw_data(Some(selection), GetOptions::default())
        .await;
    log::info!("RawArray {:?}", raw.shape());
    let d = raw.data_uint16();

    let volume = make_raw_volume_block(d, raw.shape());

    let html_canvas = canvas
        .clone()
        .unchecked_into::<web_sys::HtmlCanvasElement>();
    let html_canvas2 = html_canvas.clone();

    let builder = window_builder_without_size("Offscreen DVR".to_string(), html_canvas);
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let window = builder.build(&event_loop).unwrap();

    // instantiate global event proxy
    unsafe {
        GLOBAL_EVENT_LOOP_PROXY = Some(event_loop.create_proxy());
    }

    register_default_js_event_handlers(&html_canvas2, &event_loop);

    // this part shows a GPUDevice handle can be shared via a custom event posted to the canvas
    let exposed_device = expose_device().await;
    let event_from_rust = web_sys::CustomEvent::new("from-rust").ok().unwrap();
    event_from_rust.init_custom_event_with_can_bubble_and_cancelable_and_detail(
        "from-rust",
        false,
        false,
        &exposed_device,
    );
    html_canvas2.dispatch_event(&event_from_rust).ok();

    let mut loader_request = HashMap::new();
    loader_request.insert("store", "http://localhost:8005/");
    loader_request.insert("path", "ome-zarr/m.ome.zarr/0");
    event_from_rust.init_custom_event_with_can_bubble_and_cancelable_and_detail(
        "loader-request",
        false,
        false,
        &JsValue::from_serde(&loader_request).unwrap(),
    );
    html_canvas2.dispatch_event(&event_from_rust).ok();

    let renderer = Renderer::new(canvas, volume).await;

    let start_closure = Closure::once_into_js(move || run_event_loop(renderer, window, event_loop));

    // make sure to handle JS exceptions thrown inside start.
    // Otherwise wasm_bindgen_futures Queue would break and never handle any tasks again.
    // This is required, because winit uses JS exception for control flow to escape from `run`.
    if let Err(error) = call_catch(&start_closure) {
        let is_control_flow_exception = error.dyn_ref::<js_sys::Error>().map_or(false, |e| {
            e.message().includes("Using exceptions for control flow", 0)
        });

        if !is_control_flow_exception {
            web_sys::console::error_1(&error);
        }
    }

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(catch, js_namespace = Function, js_name = "prototype.call.call")]
        fn call_catch(this: &JsValue) -> Result<(), JsValue>;
    }
}

pub fn run_event_loop(renderer: Renderer, window: Window, event_loop: EventLoop<Event<()>>) {
    // TODO: refactor these params
    let distance_from_center = 50.;

    let resolution = Vec2::new(
        renderer.canvas.width() as f32,
        renderer.canvas.height() as f32,
    );

    const TRANSLATION_SPEED: f32 = 5.0;

    const NEAR: f32 = 0.0001;
    const FAR: f32 = 1000.0;
    let perspective = Projection::new_perspective(
        f32::to_radians(45.),
        renderer.canvas.width() as f32 / renderer.canvas.height() as f32,
        NEAR,
        FAR,
    );
    let orthographic = Projection::new_orthographic(Bounds3D::new(
        (resolution * -0.5).extend(NEAR),
        (resolution * 0.5).extend(FAR),
    ));

    let mut camera = Camera::new(
        CameraView::new(
            Vec3::new(1., 1., 1.) * distance_from_center,
            Vec3::new(0., 0., 0.),
            Vec3::new(0., 1., 0.),
        ),
        perspective.clone(),
    );
    let mut last_mouse_position = Vec2::new(0., 0.);
    let mut left_mouse_pressed = false;
    let mut right_mouse_pressed = false;

    event_loop.run(move |event, _, control_flow| {
        // force ownership by the closure
        let _ = (&renderer.ctx.instance, &renderer.ctx.adapter);

        *control_flow = winit::event_loop::ControlFlow::Poll;

        // todo: refactor input handling
        match event {
            winit::event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            // todo: handle events
            winit::event::Event::UserEvent(e) => match e {
                event::Event::Window(window_event) => match window_event {
                    WindowEvent::Resized(_) => {}
                    WindowEvent::Moved(_) => {}
                    WindowEvent::CloseRequested => {}
                    WindowEvent::Destroyed => {}
                    WindowEvent::DroppedFile(_) => {}
                    WindowEvent::HoveredFile(_) => {}
                    WindowEvent::HoveredFileCancelled => {}
                    WindowEvent::ReceivedCharacter(_) => {}
                    WindowEvent::Focused(_) => {}
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(virtual_keycode),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => match virtual_keycode {
                        VirtualKeyCode::D => camera.view.move_right(TRANSLATION_SPEED),
                        VirtualKeyCode::A => camera.view.move_left(TRANSLATION_SPEED),
                        VirtualKeyCode::W | VirtualKeyCode::Up => {
                            camera.view.move_forward(TRANSLATION_SPEED)
                        }
                        VirtualKeyCode::S | VirtualKeyCode::Down => {
                            camera.view.move_backward(TRANSLATION_SPEED)
                        }
                        VirtualKeyCode::C => {
                            if camera.projection().is_orthographic() {
                                camera.set_projection(perspective.clone());
                            } else {
                                camera.set_projection(orthographic.clone());
                            }
                        }
                        _ => {}
                    },
                    WindowEvent::ModifiersChanged(_) => {}
                    WindowEvent::CursorMoved { position, .. } => {
                        let mouse_position = glam::Vec2::new(position.x as f32, position.y as f32);
                        let delta = (mouse_position - last_mouse_position) / resolution;
                        last_mouse_position = mouse_position;

                        if left_mouse_pressed {
                            camera.view.orbit(delta, false);
                        } else if right_mouse_pressed {
                            let translation = delta * TRANSLATION_SPEED * 20.;
                            camera.view.move_right(translation.x);
                            camera.view.move_down(translation.y);
                        }
                    }
                    WindowEvent::CursorEntered { .. } => {}
                    WindowEvent::CursorLeft { .. } => {}
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::PixelDelta(delta),
                        ..
                    } => {
                        camera.view.move_forward(
                            (f64::min(delta.y.abs(), 1.) * delta.y.signum()) as f32
                                * TRANSLATION_SPEED,
                        );
                    }
                    WindowEvent::MouseInput { state, button, .. } => match button {
                        MouseButton::Left => {
                            left_mouse_pressed = state == ElementState::Pressed;
                        }
                        MouseButton::Right => {
                            right_mouse_pressed = state == ElementState::Pressed;
                        }
                        _ => {}
                    },
                    WindowEvent::TouchpadPressure { .. } => {}
                    WindowEvent::AxisMotion { .. } => {}
                    WindowEvent::Touch(_) => {}
                    WindowEvent::ScaleFactorChanged { .. } => {}
                    WindowEvent::ThemeChanged(_) => {}
                    _ => {}
                },
                Event::RawArray(raw_array) => {
                    log::info!("got raw array {:?}", raw_array.shape);
                }
                _ => {}
            },
            winit::event::Event::RedrawRequested(_) => {
                renderer.update(&camera);

                let frame = match renderer.ctx.surface.as_ref().unwrap().get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        renderer.ctx.surface.as_ref().unwrap().configure(
                            &renderer.ctx.device,
                            renderer.ctx.surface_configuration.as_ref().unwrap(),
                        );
                        renderer
                            .ctx
                            .surface
                            .as_ref()
                            .unwrap()
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                renderer.render(&view);

                frame.present();
            }
            _ => {}
        }
    });
}

pub fn make_raw_volume_block(data: Vec<u16>, shape: Vec<u32>) -> RawVolumeBlock {
    let volume_max = *data.iter().max().unwrap() as f32;
    RawVolumeBlock::new(
        data.iter()
            .map(|x| ((*x as f32 / volume_max) * 255.) as u8)
            .collect(),
        volume_max as u32,
        shape[2],
        shape[1],
        shape[0],
    )
}

#[wasm_bindgen(js_name = "maxValue")]
pub fn max_value(data: Vec<u16>) -> f32 {
    *data.par_iter()
        .max()
        .unwrap() as f32
}

#[wasm_bindgen(js_name = "convertToUint8")]
pub fn convert_to_uint8(data: Vec<u16>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|x| ((*x as f32 / max_value) * 255.) as u8)
        .collect()
}

// Test device sharing between WASM and JS context, could be useful at some point

#[wasm_bindgen(module = "/shared-gpu.js")]
extern "C" {
    #[wasm_bindgen(js_name = "printGpuDeviceLimits")]
    fn print_gpu_device_limits(device: web_sys::GpuDevice);
}

#[wasm_bindgen(js_name = "testDeviceSharing")]
pub fn test_device_sharing() {
    wasm_bindgen_futures::spawn_local(get_device());
}

async fn get_device() {
    print_gpu_device_limits(expose_device().await);
}

async fn expose_device() -> web_sys::GpuDevice {
    // create ctx to capture device from
    let mut ctx =
        renderer::context::GPUContext::new(&renderer::context::ContextDescriptor::default()).await;

    // memcopy device
    //let device = transmute_copy!(ctx.device, Device);
    //let device = util::resource::copy_device(&ctx.device);

    let device = util::transmute::transmute_copy!(ctx.device, util::transmute::Device);

    // make sure ctx still has its device
    log::info!("max bind groups: {}", ctx.device.limits().max_bind_groups);

    device.id
}
