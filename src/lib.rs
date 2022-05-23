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
    ome_ngff::foo();
}

// todo: test https://rustwasm.github.io/wasm-bindgen/reference/arbitrary-data-with-serde.html

#[wasm_bindgen(js_name="sendExampleToJS")]
pub fn send_example_to_js() -> JsValue {
    let mut field1 = HashMap::new();
    field1.insert(0, String::from("eaasdfasdfasdfasdfx"));
    let example = ome_ngff::Example {
        field1,
        field2: vec![vec![1., 2.], vec![3., 4.]],
        field3: [1., 2., 3., 4.],
        axes: vec![
            ome_ngff::Axis::Space(ome_ngff::axes::SpaceAxis::new(
                "foo".to_string(),
                Some(ome_ngff::SpaceUnit::Angstrom)
            )),
            ome_ngff::Axis::Time(ome_ngff::axes::TimeAxis::new(
                "foo".to_string(),
                Some(ome_ngff::TimeUnit::Attosecond)
            )),
            ome_ngff::Axis::Channel(ome_ngff::axes::ChannelAxis::new(
                "foo".to_string(),
            )),
            ome_ngff::Axis::Custom(ome_ngff::axes::CustomAxis::new(
                "foo".to_string(),
                Some("lalala".to_string()),
                Some("unit".to_string()),
            ))
        ],
    };

    JsValue::from_serde(&example).unwrap()
}

#[wasm_bindgen(js_name = "receiveExampleFromJS")]
pub fn receive_example_from_js(val: &JsValue) -> JsValue {
    let mut example: ome_ngff::Example = val.into_serde().unwrap();
    //example.axis.name = "lalala".parse().unwrap();
    JsValue::from_serde(&example).unwrap()
}
