extern crate core;

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Vec2, Vec3};
use wasm_bindgen::{prelude::*, JsCast};
use winit::event::{
    ElementState, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode, WindowEvent,
};
use winit::event_loop::EventLoop;
use winit::platform::web::EventLoopExtWebSys;
use winit::window::Window;

pub use numcodecs_wasm::*;
pub use zarr_wasm::zarr::{DimensionArraySelection, GetOptions, ZarrArray};

pub mod app;
pub mod event;
pub mod gpu_list;
pub mod input;
pub mod preprocessing;
pub mod renderer;
pub mod resource;
pub mod util;
pub mod volume;
pub mod wgsl;

use crate::event::{ChannelSettingsChange, Event, RawArrayReceived, SettingsChange};
use util::init;
use util::window;

use crate::event::handler::register_default_js_event_handlers;
use crate::input::Input;
use crate::renderer::camera::{Camera, CameraView, Projection};
use crate::renderer::context::GPUContext;
use crate::renderer::geometry::Bounds3D;
use crate::renderer::settings::MultiChannelVolumeRendererSettings;
use crate::renderer::volume::RawVolumeBlock;
use crate::renderer::MultiChannelVolumeRenderer;
use crate::resource::SparseResidencyTexture3D;
use crate::volume::{
    BrickedMultiResolutionMultiVolumeMeta, HtmlEventTargetVolumeDataSource, VolumeDataSource,
};
use crate::window::window_builder_without_size;

// todo: remove this (this is for testing the preprocessor macro)
use crate::util::vec::vec_equals;
#[allow(unused)]
use include_preprocessed_wgsl::include_preprocessed;

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
pub fn main(canvas: JsValue, volume_meta: JsValue, render_settings: JsValue) {
    // todo: make logger configurable
    init::set_panic_hook();
    init::set_logger(None);

    /* todo: remove this (this is for testing the preprocessor macro only)
    let _shader_str = include_preprocessed!(
        "renderer/pass/dvr/dvr.wgsl",
        include_dirs = [
            "renderer/wgsl",
            {
                path: "renderer/wgsl",
                relative: true,
            }
        ]
        includes = [
            "renderer/pass/process_requests/process_requests.wgsl",
            {
                identifier: "ray",
                path: "renderer/ray.wgsl"
            },
            {
                identifier: "ray",
                path: {
                    path: "/home/foo/wgsl/ray.wgsl",
                    relative: false,
                },
            }
            ["ray", "renderer/ray.wgsl"]
        ]
    );
     */

    let volume_meta: BrickedMultiResolutionMultiVolumeMeta = volume_meta
        .into_serde()
        .expect("Received invalid volume meta. Shutting down.");

    let render_settings: MultiChannelVolumeRendererSettings = render_settings
        .into_serde()
        .expect("Received invalid render settings. Shutting down.");

    wasm_bindgen_futures::spawn_local(start_event_loop(canvas, volume_meta, render_settings));
}

async fn start_event_loop(
    canvas: JsValue,
    volume_meta: BrickedMultiResolutionMultiVolumeMeta,
    render_settings: MultiChannelVolumeRendererSettings,
) {
    let html_canvas = canvas
        .clone()
        .unchecked_into::<web_sys::HtmlCanvasElement>();

    let builder = window_builder_without_size("Offscreen DVR".to_string(), html_canvas.clone());
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let window = builder.build(&event_loop).unwrap();

    // instantiate global event proxy
    unsafe {
        GLOBAL_EVENT_LOOP_PROXY = Some(event_loop.create_proxy());
    }

    register_default_js_event_handlers(&html_canvas, &event_loop);

    // this part shows a GPUDevice handle can be shared via a custom event posted to the canvas
    let exposed_device = expose_device().await;
    let event_from_rust = web_sys::CustomEvent::new("from-rust").ok().unwrap();
    event_from_rust.init_custom_event_with_can_bubble_and_cancelable_and_detail(
        "from-rust",
        false,
        false,
        &exposed_device,
    );
    html_canvas.dispatch_event(&event_from_rust).ok();

    let renderer = MultiChannelVolumeRenderer::new(
        canvas,
        Box::new(HtmlEventTargetVolumeDataSource::new(
            volume_meta,
            html_canvas.clone().unchecked_into::<web_sys::EventTarget>(),
        )),
        &render_settings,
    )
    .await;

    // NOTE: All resource allocations should happen before the main render loop
    // The reason for this is that receiving allocation errors is async, but
    let start_closure = Closure::once_into_js(move || {
        run_event_loop(renderer, render_settings, window, event_loop)
    });

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

pub fn run_event_loop(
    renderer: MultiChannelVolumeRenderer,
    render_settings: MultiChannelVolumeRendererSettings,
    window: Window,
    event_loop: EventLoop<Event<()>>,
) {
    let renderer = Rc::new(RefCell::new(renderer));
    let mut settings = render_settings.clone();
    let mut last_channel_selection = settings.get_sorted_visible_channel_indices();
    let mut last_input = Input::default();

    // TODO: refactor these params
    let distance_from_center = 500.;

    let window_size = renderer.as_ref().borrow().window_size;

    let resolution = Vec2::new(window_size.width as f32, window_size.height as f32);

    const TRANSLATION_SPEED: f32 = 5.0;

    const NEAR: f32 = 0.0001;
    const FAR: f32 = 1000.0;
    let perspective = Projection::new_perspective(
        f32::to_radians(45.),
        window_size.width as f32 / window_size.height as f32,
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
        perspective,
    );
    let mut last_mouse_position = Vec2::new(0., 0.);
    let mut left_mouse_pressed = false;
    let mut right_mouse_pressed = false;

    let window = Rc::new(window);

    event_loop.spawn(move |event, _, control_flow| {
        // force ownership by the closure
        //let _ = (&renderer.as_ref().borrow().ctx.instance, &renderer.as_ref().borrow().ctx.adapter);

        *control_flow = winit::event_loop::ControlFlow::Poll;

        // todo: refactor input handling
        match event {
            winit::event::Event::RedrawEventsCleared => {
                //window.request_redraw();
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
                                camera.set_projection(perspective);
                            } else {
                                camera.set_projection(orthographic);
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
                Event::Settings(settings_change) => match settings_change {
                    SettingsChange::RenderMode(mode) => {
                        settings.render_mode = mode;
                    }
                    SettingsChange::StepScale(step_scale) => {
                        if step_scale > 0. {
                            settings.step_scale = step_scale;
                        } else {
                            log::error!("Illegal step size: {}", step_scale);
                        }
                    }
                    SettingsChange::MaxSteps(max_steps) => {
                        settings.max_steps = max_steps;
                    }
                    SettingsChange::BackgroundColor(color) => {
                        settings.background_color = color;
                    }
                    SettingsChange::ChannelSetting(channel_setting) => {
                        let i = channel_setting.channel_index as usize;
                        match channel_setting.channel_setting {
                            ChannelSettingsChange::Color(color) => {
                                settings.channel_settings[i].color = color;
                            }
                            ChannelSettingsChange::Visible(visible) => {
                                settings.channel_settings[i].visible = visible;
                            }
                            ChannelSettingsChange::Threshold(range) => {
                                settings.channel_settings[i].threshold_lower = range.min;
                                settings.channel_settings[i].threshold_upper = range.max;
                            }
                            ChannelSettingsChange::LoD(range) => {
                                settings.channel_settings[i].max_lod = range.min;
                                settings.channel_settings[i].min_lod = range.max;
                            }
                        }
                    }
                },
                _ => {}
            },
            // todo: refactor this
            winit::event::Event::RedrawRequested(_) => {
                let channel_selection = settings.get_sorted_visible_channel_indices();

                let input = if vec_equals(&channel_selection, &last_channel_selection) {
                    Input::from_last(&last_input)
                } else {
                    last_channel_selection = channel_selection.clone();
                    Input::from_last_with_channel_selection(&last_input, channel_selection)
                };

                renderer
                    .as_ref()
                    .borrow()
                    .update(&camera, &input, &settings);

                wasm_bindgen_futures::spawn_local(calling_from_async(
                    renderer.clone(),
                    camera,
                    input.clone(),
                    window.clone(),
                ));

                last_input = input;
            }
            _ => {}
        }
    });
}

async fn calling_from_async(
    renderer: Rc<RefCell<MultiChannelVolumeRenderer>>,
    _camera: Camera,
    input: Input,
    window: Rc<Window>,
) {
    let frame = match renderer
        .as_ref()
        .borrow()
        .ctx
        .surface
        .as_ref()
        .unwrap()
        .get_current_texture()
    {
        Ok(frame) => frame,
        Err(_) => {
            renderer
                .as_ref()
                .borrow()
                .ctx
                .surface
                .as_ref()
                .unwrap()
                .configure(
                    &renderer.as_ref().borrow().ctx.device,
                    renderer
                        .as_ref()
                        .borrow()
                        .ctx
                        .surface_configuration
                        .as_ref()
                        .unwrap(),
                );
            renderer
                .as_ref()
                .borrow()
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

    renderer.as_ref().borrow().render(&view, &input);

    renderer.as_ref().borrow_mut().post_render(&input);

    frame.present();

    // we request the redraw after rendering has definitely finished
    window.request_redraw();
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

// Test device sharing between WASM and JS context, could be useful at some point

#[wasm_bindgen(module = "/js/src/shared-gpu.js")]
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
    let mut ctx = GPUContext::new(&renderer::context::ContextDescriptor::default()).await;

    // memcopy device
    //let device = transmute_copy!(ctx.device, Device);
    //let device = util::resource::copy_device(&ctx.device);

    let device = util::transmute::transmute_copy!(ctx.device, util::transmute::Device);

    // make sure ctx still has its device
    log::info!("max bind groups: {}", ctx.device.limits().max_bind_groups);

    device.id
}
