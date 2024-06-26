use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "https://cdn.skypack.dev/numcodecs")]
extern "C" {
    pub type Blosc;

    #[wasm_bindgen(constructor)]
    fn new() -> Blosc;

    #[wasm_bindgen(js_namespace = Blosc, js_name = "fromConfig")]
    fn from_config(config: &JsValue) -> Blosc;

    //#[wasm_bindgen(method)]
    //fn encode();

    //#[wasm_bindgen(method)]
    //fn decode();
}

#[wasm_bindgen(js_name = "makeBlosc")]
pub fn make_blosc(config: &JsValue) -> Blosc {
    Blosc::from_config(config)
}
