
use std::path::PathBuf;
use std::str::FromStr;
use bevy_window::{WindowDescriptor, WindowId};

use serde_json;

mod util;

use util::init;
use util::io;

use wasm_bindgen::prelude::*;
use instant;

pub use numcodecs_wasm::*;

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

// starts an event loop
#[wasm_bindgen]
pub fn main() {

}

// playground stuff

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

// todo: check out how to use typescript libraries from rust and decode zarr arrays
// todo: goal on monday is to read my OME-Zarr structure in rust
//  next goal is to put it into a GPU buffer and render it


#[wasm_bindgen(js_name="lala")]
pub fn greet() {
    log::info!("Logging that mofo!!!!!!! {}", instant::now());
    use wgpu;
    wgpu::Instance::new(wgpu::Backends::all());
}

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
