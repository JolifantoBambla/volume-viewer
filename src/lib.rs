mod util;

use std::collections::HashMap;
use util::init;

use wasm_bindgen::prelude::*;
use instant;

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

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen(js_name="lala")]
pub fn greet() {
    log::info!("Logging that mofo {}", instant::now());
}

// todo: test https://rustwasm.github.io/wasm-bindgen/reference/arbitrary-data-with-serde.html

#[wasm_bindgen(js_name="sendExampleToJS")]
pub fn send_example_to_js() -> JsValue {
    let multiscale = ome_ngff::multiscale::Multiscale {
        name: Some("foo".to_string()),
        version: Some("0.4.0".to_string()),
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
            ome_ngff::multiscale::Dataset{
                path: "0".to_string(),
                coordinate_transformations: vec![
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Scale(ome_ngff::coordinate_transformations::Scale::Scale(vec![1.0, 2.0, 3.0])),
                    ome_ngff::coordinate_transformations::CoordinateTransformation::Translation(ome_ngff::coordinate_transformations::Translation::Translation(vec![1.0, 2.0, 3.0])),
                ]
            },
            ome_ngff::multiscale::Dataset{
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
    log::info!("multiscale valid (expect false): {}", example.is_valid());
    JsValue::from_serde(&example).unwrap()
}
