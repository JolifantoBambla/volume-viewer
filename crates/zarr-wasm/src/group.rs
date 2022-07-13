use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use crate::persistence_mode::PersistenceMode;

#[wasm_bindgen(module = "https://cdn.skypack.dev/zarr")]
extern "C" {
    #[wasm_bindgen(js_name = "Attributes", typescript_type = "Attributes")]
    #[derive(Debug, Clone)]
    type Attributes;

    #[wasm_bindgen(method, js_class = "Attributes", js_name = "asObject")]
    async fn as_object(this: &Attributes) -> JsValue;
}

#[wasm_bindgen(module = "https://cdn.skypack.dev/zarr")]
extern "C" {
    #[wasm_bindgen(js_name = "Group", typescript_type = "Group")]
    #[derive(Debug, Clone)]
    pub type Group;

    #[wasm_bindgen(method, getter, js_class = "Group")]
    pub fn name(this: &Group) -> String;

    #[wasm_bindgen(method, getter, js_class = "Group")]
    pub fn basename(this: &Group) -> String;

    #[wasm_bindgen(method, getter, js_class = "Group")]
    fn attrs(this: &Group) -> JsValue;

    #[wasm_bindgen(js_name = "openGroup")]
    async fn open_group(store: String, path: String, mode: JsValue, chunk_store: JsValue, cache_attrs: bool) -> JsValue;
}

impl Group {
    pub async fn open(store: String, path: String) -> Group {
        let mode = JsValue::from_serde(&PersistenceMode::ReadOnly).unwrap();
        let group = open_group(store, path, mode, JsValue::NULL, true).await;
        group.unchecked_into()
    }

    pub async fn get_attributes(&self) -> ome_ngff::metadata::Metadata {
        self.attrs()
            .unchecked_into::<Attributes>()
            .as_object()
            .await
            .into_serde()
            .expect("Failed to deserialize attributes")
    }
}
