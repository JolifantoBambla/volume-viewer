use serde::{Serialize, Deserialize};

use wasm_bindgen::{prelude::*, JsCast};

pub use numcodecs_wasm::*;

pub mod util;
pub mod renderer;

use util::init;
use util::window;

use crate::renderer::playground::{RawVolume, ZSlicer};
use crate::renderer::volume::RawVolumeBlock;

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

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub canvas: Option<String>,
}

// starts an event loop using bevy stuff
#[wasm_bindgen]
pub fn main(js_config: &JsValue) {
    let context_descriptor = renderer::context::ContextDescriptor::default();
    log::info!("contextdesscritpor {}", context_descriptor.backends.bits());

    let config: Config = js_config.into_serde().unwrap();

    fn really_annoying_back_seat_kid() {
        log::debug!("Are we there yet? It's already {}", instant::now());
    }

    bevy_app::App::new()
        .insert_resource(bevy_window::WindowDescriptor{
            title: "I am a window!".to_string(),
            width: 500.,
            height: 300.,
            canvas: config.canvas,
            ..bevy_window::WindowDescriptor::default()
        })
        .add_plugin(bevy_window::WindowPlugin::default())
        .add_plugin(bevy_winit::WinitPlugin::default())
        .add_plugin(bevy_app::ScheduleRunnerPlugin::default())
        .add_system_to_stage(bevy_app::CoreStage::Update, really_annoying_back_seat_kid)
        .run()
}

#[wasm_bindgen(js_name = "runComputeExample")]
pub fn run_compute_example() {
    wasm_bindgen_futures::spawn_local(compute_example());
}

async fn compute_example() {
    let (window, _) = window::create_window("compute example".to_string(), "existing-canvas".to_string());

    log::info!("running compute");
    renderer::playground::compute_to_image_test(&window).await;
    log::info!("ran compute");
}

#[allow(dead_code)]
enum Examples {
    ZSlicer,
    Dvr,
}

#[wasm_bindgen(js_name = "runVolumeExample")]
pub fn run_volume_example(data: &[u16], shape: &[u32]) {
    wasm_bindgen_futures::spawn_local(volume_example(data.to_vec(), shape.to_vec(), Examples::Dvr));
}

async fn make_z_slicer_example(data: Vec<u16>, shape: Vec<u32>, window: winit::window::Window, event_loop: winit::event_loop::EventLoop<()>) -> JsValue {
    let z_slicer = renderer::playground::ZSlicer::new(
        window,
        RawVolume{
            data: data.iter().map(|x| *x as u32).collect(),
            shape,
        },
        "z-slice".to_string(),
    ).await;
    Closure::once_into_js(move || ZSlicer::run(z_slicer, event_loop))
}

async fn make_dvr_example(data: Vec<u16>, shape: Vec<u32>, window: winit::window::Window, event_loop: winit::event_loop::EventLoop<()>) -> JsValue {
    // preprocess volume
    let volume_max = *data.iter().max().unwrap() as f32;
    let volume = RawVolumeBlock::new(
        data.iter().map(|x| ((*x as f32 / volume_max) * 255.) as u8).collect(),
        volume_max as u32,
        shape[2],
        shape[1],
        shape[0],
    );

    let dvr = renderer::dvr_playground::DVR::new(
        window,
        volume,
    ).await;
    Closure::once_into_js(move || renderer::dvr_playground::DVR::run(dvr, event_loop))
}

async fn volume_example(data: Vec<u16>, shape: Vec<u32>, example: Examples) {
    let (window, event_loop) = window::create_window("compute example".to_string(), "existing-canvas".to_string());

    let start_closure = match example {
        Examples::ZSlicer => {
            make_z_slicer_example(data, shape, window, event_loop).await
        }
        Examples::Dvr => {
            make_dvr_example(data, shape, window, event_loop).await
        }
    };

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

// Test device sharing between WASM and JS context, could be useful at some point

#[wasm_bindgen(module = "/shared-gpu.js")]
extern {
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
    let mut ctx = renderer::context::GPUContext::new(&renderer::context::ContextDescriptor::default(), None).await;

    // memcopy device
    //let device = transmute_copy!(ctx.device, Device);
    //let device = util::resource::copy_device(&ctx.device);

    let device = util::transmute::transmute_copy!(ctx.device, util::transmute::Device);

    // make sure ctx still has its device
    log::info!("max bind groups: {}", ctx.device.limits().max_bind_groups);

    device.id
}
