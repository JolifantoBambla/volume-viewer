use std::any::Any;
use std::convert::{Infallible, TryFrom};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use instant;
use serde::{Serialize, Deserialize};
use serde_json;

use wasm_bindgen::{prelude::*, JsCast};

pub use numcodecs_wasm::*;

pub mod util;
pub mod renderer;

use util::init;
use util::io;
use util::window;

use crate::renderer::GPUContext;

use web_sys;
use crate::renderer::playground::{RawVolume, ZSlicer};

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
    let context_descriptor = renderer::ContextDescriptor::default();
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
    let (window, event_loop) = window::create_window("compute example".to_string(), "existing-canvas".to_string());

    log::info!("running compute");
    renderer::playground::compute_to_image_test(&window).await;
    log::info!("ran compute");
}


#[wasm_bindgen(js_name = "runVolumeExample")]
pub fn run_volume_example(data: &[u8], shape: &[u32]) {
    wasm_bindgen_futures::spawn_local(volume_example(data.to_vec(), shape.to_vec()));
}

async fn volume_example(data: Vec<u8>, shape: Vec<u32>) {
    let (window, event_loop) = window::create_window("compute example".to_string(), "existing-canvas".to_string());

    log::info!("creating example");
    let z_slicer = renderer::playground::ZSlicer::new(
        window,
        RawVolume{
            data,
            shape,
        },
        "z-slice".to_string(),
    ).await;
    log::info!("starting example");
    let start_closure = Closure::once_into_js(move || ZSlicer::run(z_slicer, event_loop));

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
    // helper structs to extract private fields
    struct Context(web_sys::Gpu);
    #[derive(Clone)]
    struct Device {
        context: Arc<Context>,
        pub id: web_sys::GpuDevice,
    }

    // create ctx to capture device from
    let mut ctx = renderer::GPUContext::new(&renderer::ContextDescriptor::default(), None).await;

    // memcopy device
    //let device = transmute_copy!(ctx.device, Device);
    //let device = util::resource::copy_device(&ctx.device);

    let device = util::transmute::transmute_copy!(ctx.device, util::transmute::Device);

    // make sure ctx still has its device
    log::info!("max bind groups: {}", ctx.device.limits().max_bind_groups);

    device.id
}

#[wasm_bindgen(js_name = "printZSlice")]
pub fn print_z_slice(data: &[u16]) {
    log::info!("data: {:?}", data);
}

// playground stuff

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

// todo: check out how to use typescript libraries from rust and decode zarr arrays
// todo: goal on monday is to read my OME-Zarr structure in rust
//  next goal is to put it into a GPU buffer and render it

// todo: test https://rustwasm.github.io/wasm-bindgen/reference/arbitrary-data-with-serde.html

#[wasm_bindgen(js_name="sendExampleToJS")]
pub fn send_example_to_js() -> JsValue {
    let multiscale = ome_ngff::multiscale::v0_4::Multiscale {
        name: Some("foo".to_string()),
        downscaling_type: Some("gaussian".to_string()),
        axes: vec![
            ome_ngff::axis::Axis::Space(ome_ngff::axis::SpaceAxis{
                name: "foo".to_string(),
                unit: Some(ome_ngff::axis::SpaceUnit::Angstrom)
            }),
            ome_ngff::axis::Axis::Time(ome_ngff::axis::TimeAxis{
                name: "foo".to_string(),
                unit: Some(ome_ngff::axis::TimeUnit::Attosecond)
            }),
            ome_ngff::axis::Axis::Channel(ome_ngff::axis::ChannelAxis{
                name: "foo".to_string(),
            }),
            ome_ngff::axis::Axis::Custom(ome_ngff::axis::CustomAxis {
                name: "foo".to_string(),
                axis_type: Some("lalala".to_string()),
                unit: Some("unit".to_string()),
            })
        ],
        datasets: vec![
            ome_ngff::multiscale::v0_4::Dataset{
                path: "0".to_string(),
                coordinate_transformations: vec![
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Scale(ome_ngff::coordinate_transformations::Scale::Scale(vec![1.0, 2.0, 3.0])),
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Translation(ome_ngff::coordinate_transformations::Translation::Translation(vec![1.0, 2.0, 3.0])),
                ]
            },
            ome_ngff::multiscale::v0_4::Dataset{
                path: "1".to_string(),
                coordinate_transformations: vec![
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Scale(ome_ngff::coordinate_transformations::Scale::Scale(vec![1.0, 2.0, 3.0])),
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Translation(ome_ngff::coordinate_transformations::Translation::Translation(vec![1.0, 2.0, 3.0])),
                ]
            },
        ],
        coordinate_transformations: Some(vec![
            ome_ngff::coordinate_transformations::CoordinateTransformation::Identity(ome_ngff::coordinate_transformations::Identity{}),
            ome_ngff::coordinate_transformations::CoordinateTransformation::Translation(ome_ngff::coordinate_transformations::Translation::Translation(vec![1.0, 2.0, 3.0])),
            ome_ngff::coordinate_transformations::CoordinateTransformation::Translation(ome_ngff::coordinate_transformations::Translation::Path("/path/to/translation.smth".to_string())),
            ome_ngff::coordinate_transformations::CoordinateTransformation::Scale(ome_ngff::coordinate_transformations::Scale::Scale(vec![1.0, 2.0, 3.0])),
            ome_ngff::coordinate_transformations::CoordinateTransformation::Scale(ome_ngff::coordinate_transformations::Scale::Path("/path/to/translation.smth".to_string())),
        ]),
        metadata: None
    };

    log::info!("multiscale valid (expect false): {}", multiscale.is_valid());

    JsValue::from_serde(&multiscale).unwrap()
}

#[wasm_bindgen(js_name = "receiveExampleFromJS")]
pub fn receive_example_from_js(val: &JsValue) -> JsValue {
    let mut example: ome_ngff::multiscale::Multiscale = val.into_serde().unwrap();
    match &example {
        ome_ngff::multiscale::Multiscale::V0_2(m) => {
            log::info!("got a v0.2!");
        },
        ome_ngff::multiscale::Multiscale::V0_3(m) => {
            log::info!("got a v0.3!");
        }
        ome_ngff::multiscale::Multiscale::V0_4(m) => {
            log::info!("got a v0.4 and is_valid returned: {}", m.is_valid());
        },
        _ => {}
    }
    JsValue::from_serde(&example).unwrap()
}

#[wasm_bindgen(js_name = "receiveZarrayFromJS")]
pub fn receive_zarray_from_js(val: &JsValue) -> JsValue {
    let mut example: ome_zarr::zarr_v2::ArrayMetadata = val.into_serde().unwrap();
    JsValue::from_serde(&example).unwrap()
}

#[wasm_bindgen(js_name = "readMyFile")]
pub async fn read_my_file() -> JsValue {
    let foo = io::load_file_as_string(PathBuf::from_str("http://localhost:8005/ome-zarr/m.ome.zarr/0/.zattrs").unwrap())
        .await;
    let metadata: ome_ngff::metadata::Metadata = serde_json::from_str(foo.as_str()).unwrap();
    let multiscales = &metadata.multiscales.unwrap();
    let multiscale: &ome_ngff::multiscale::Multiscale = multiscales.first().unwrap();
    match &multiscale {
        ome_ngff::multiscale::Multiscale::V0_2(m) => {
            log::info!("got a v0.2!");
        },
        ome_ngff::multiscale::Multiscale::V0_3(m) => {
            log::info!("got a v0.3!");
        }
        ome_ngff::multiscale::Multiscale::V0_4(m) => {
            log::info!("got a v0.4 and is_valid returned: {}", m.is_valid());
        },
        _ => {}
    }
    JsValue::from_serde(&multiscale).unwrap()
}
