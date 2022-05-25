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
    let multiscale = ome_ngff::Multiscale {
        name: Some("foo".to_string()),
        version: Some("0.4.0".to_string()),
        downscaling_type: Some("gaussian".to_string()),
        axes: vec![
            ome_ngff::Axis::Space(ome_ngff::axis::SpaceAxis{
                name: "foo".to_string(),
                unit: Some(ome_ngff::SpaceUnit::Angstrom)
            }),
            ome_ngff::Axis::Time(ome_ngff::axis::TimeAxis{
                name: "foo".to_string(),
                unit: Some(ome_ngff::TimeUnit::Attosecond)
            }),
            ome_ngff::Axis::Channel(ome_ngff::axis::ChannelAxis{
                name: "foo".to_string(),
            }),
            ome_ngff::Axis::Custom(ome_ngff::axis::CustomAxis {
                name: "foo".to_string(),
                axis_type: Some("lalala".to_string()),
                unit: Some("unit".to_string()),
            })
        ],
        datasets: vec![
            ome_ngff::Dataset{
                path: "0".to_string(),
                coordinate_transformations: vec![
                    ome_ngff::CoordinateTransformation::Scale(ome_ngff::Scale::Scale(vec![1.0, 2.0, 3.0])),
                    ome_ngff::CoordinateTransformation::Translation(ome_ngff::Translation::Translation(vec![1.0, 2.0, 3.0])),
                ]
            },
            ome_ngff::Dataset{
                path: "1".to_string(),
                coordinate_transformations: vec![
                    ome_ngff::CoordinateTransformation::Scale(ome_ngff::Scale::Scale(vec![1.0, 2.0, 3.0])),
                    ome_ngff::CoordinateTransformation::Translation(ome_ngff::Translation::Translation(vec![1.0, 2.0, 3.0])),
                ]
            },
        ],
        coordinate_transformations: Some(vec![
            ome_ngff::CoordinateTransformation::Identity(ome_ngff::Identity{}),
            ome_ngff::CoordinateTransformation::Translation(ome_ngff::Translation::Translation(vec![1.0, 2.0, 3.0])),
            ome_ngff::CoordinateTransformation::Translation(ome_ngff::Translation::Path("/path/to/translation.smth".to_string())),
            ome_ngff::CoordinateTransformation::Scale(ome_ngff::Scale::Scale(vec![1.0, 2.0, 3.0])),
            ome_ngff::CoordinateTransformation::Scale(ome_ngff::Scale::Path("/path/to/translation.smth".to_string())),
        ]),
        metadata: None
    };

    JsValue::from_serde(&multiscale).unwrap()
}

#[wasm_bindgen(js_name = "receiveExampleFromJS")]
pub fn receive_example_from_js(val: &JsValue) -> JsValue {
    let mut example: ome_ngff::Multiscale = val.into_serde().unwrap();
    //example.axis.name = "lalala".parse().unwrap();
    JsValue::from_serde(&example).unwrap()
}
