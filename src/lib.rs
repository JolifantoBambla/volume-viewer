use wasm_bindgen::{prelude::*, JsCast};

pub use numcodecs_wasm::*;

pub mod event;
pub mod renderer;
pub mod util;

use util::init;
use util::window;

use crate::event::handler::register_default_js_event_handlers;
use crate::renderer::volume::RawVolumeBlock;
use crate::window::window_builder_without_size;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen(start)]
pub fn initialize() {
    // initialize panic hook
    init::set_panic_hook();

    // initialize logger
    init::set_logger(None);
}

#[wasm_bindgen(js_name = "runVolumeExample")]
pub fn run_volume_example(data: &[u16], shape: &[u32]) {
    wasm_bindgen_futures::spawn_local(volume_example(data.to_vec(), shape.to_vec()));
}

#[wasm_bindgen(js_name = "runOffscreenExample")]
pub fn run_offscreen_example(data: &[u16], shape: &[u32], canvas: JsValue) {
    wasm_bindgen_futures::spawn_local(offscreen_example(data.to_vec(), shape.to_vec(), canvas));
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

// todo: lib.rs should contain only init and event_loop stuff
//  communication with JS side should only be done via events
//  create CustomEvent enum that consists of WindowEvent and can be extended by other events
//  create thread pool that is supplied with event proxies sending events to event loop

async fn make_offscreen_dvr(data: Vec<u16>, shape: Vec<u32>, canvas: JsValue) -> JsValue {
    let volume = make_raw_volume_block(data, shape);

    let html_canvas = canvas
        .clone()
        .unchecked_into::<web_sys::HtmlCanvasElement>();
    let html_canvas2 = html_canvas.clone();

    let builder = window_builder_without_size("Offscreen DVR".to_string(), html_canvas);
    let event_loop = winit::event_loop::EventLoop::with_user_event();
    let window = builder.build(&event_loop).unwrap();

    register_default_js_event_handlers(&html_canvas2, &event_loop);

    let dvr = renderer::offscreen_playground::DVR::new(canvas, volume).await;
    Closure::once_into_js(move || renderer::offscreen_playground::DVR::run(dvr, window, event_loop))
}

async fn make_dvr_example(
    data: Vec<u16>,
    shape: Vec<u32>,
    window: winit::window::Window,
    event_loop: winit::event_loop::EventLoop<()>,
) -> JsValue {
    shared_worker().await;

    let volume = make_raw_volume_block(data, shape);

    let dvr = renderer::dvr_playground::DVR::new(window, volume).await;
    Closure::once_into_js(move || renderer::dvr_playground::DVR::run(dvr, event_loop))
}

async fn volume_example(data: Vec<u16>, shape: Vec<u32>) {
    let (window, event_loop) =
        window::create_window("compute example".to_string(), "existing-canvas".to_string());

    let start_closure = make_dvr_example(data, shape, window, event_loop).await;

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

async fn offscreen_example(data: Vec<u16>, shape: Vec<u32>, canvas: JsValue) {
    log::info!("making offscreen dvr");
    let start_closure = make_offscreen_dvr(data, shape, canvas).await;
    log::info!("starting offscreen dvr");

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

#[wasm_bindgen(module = "/loader-interface.js")]
extern "C" {
    #[wasm_bindgen(js_name = "sharedWorker")]
    async fn shared_worker() -> JsValue;
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
