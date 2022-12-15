use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::data_type::DataType;
use crate::persistence_mode::PersistenceMode;
use crate::raw::RawArray;

// TODO: mapping from zarr-wasm to ome-zarr

pub type ChunksArgument = Vec<u32>;

#[derive(Deserialize, Serialize)]
pub struct CompressorConfig {
    pub id: String,
}

#[derive(Deserialize, Serialize)]
pub enum Order {
    C,
    F,
}

pub type Store = String;

#[derive(Deserialize, Serialize)]
pub struct Filter {
    pub id: String,
}

#[derive(Deserialize, Serialize)]
pub enum DimensionSeparator {
    #[serde(rename = ".")]
    Dot,

    #[serde(rename = "/")]
    Slash,
}

impl Default for DimensionSeparator {
    fn default() -> Self {
        DimensionSeparator::Dot
    }
}

#[derive(Serialize, Deserialize)]
pub struct OpenArrayOptions {
    pub shape: Option<Vec<u32>>,
    pub mode: PersistenceMode,
    pub chunks: Option<ChunksArgument>,

    #[serde(rename = "dtype")]
    pub data_type: DataType,

    pub compressor: Option<CompressorConfig>,
    pub fill_value: Option<f32>,
    pub order: Option<Order>,
    pub store: Store,
    pub overwrite: Option<bool>,
    pub path: Option<String>,

    #[serde(rename = "chunkStore")]
    pub chunk_store: Option<Store>,
    pub filters: Option<Vec<Filter>>,

    #[serde(rename = "cacheMetadata")]
    pub cache_metadata: Option<bool>,

    #[serde(rename = "cacheAttrs")]
    pub cache_attributes: Option<bool>,

    #[serde(rename = "readOnly")]
    pub read_only: Option<bool>,

    #[serde(rename = "dimensionSeparator")]
    pub dimension_separator: Option<DimensionSeparator>,
}

impl Default for OpenArrayOptions {
    fn default() -> Self {
        Self {
            shape: None,
            mode: PersistenceMode::ReadOnly,
            chunks: None,
            data_type: DataType::Int32LittleEndian,
            compressor: None,
            fill_value: None,
            order: Some(Order::C),
            store: "".to_string(),
            overwrite: Some(false),
            path: None,
            chunk_store: None,
            filters: None,
            cache_metadata: None,
            cache_attributes: None,
            read_only: None,
            dimension_separator: None,
        }
    }
}

#[derive(Deserialize, Serialize)]
pub struct Slice {
    pub start: Option<u32>,
    pub stop: Option<u32>,
    pub step: Option<u32>,
    #[serde(rename = "_slice")]
    pub slice: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub enum DimensionArraySelection {
    Slice(Slice),
    Number(f32),
    #[serde(rename = "...")]
    Ellipsis,
    #[serde(rename = ":")]
    Separator,
}

#[derive(Deserialize, Serialize)]
pub struct GetOptions {
    #[serde(rename = "concurrencyLimit")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency_limit: Option<f32>,

    // todo: actually support this https://rustwasm.github.io/wasm-bindgen/reference/passing-rust-closures-to-js.html
    #[serde(rename = "progressCallback")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_callback: Option<String>,
}

impl Default for GetOptions {
    fn default() -> Self {
        Self {
            concurrency_limit: None,
            progress_callback: None,
        }
    }
}

#[wasm_bindgen(module = "https://cdn.skypack.dev/zarr")]
extern "C" {
    #[wasm_bindgen(js_name = "ZarrArray", typescript_type = "ZarrArray")]
    #[derive(Debug, Clone)]
    pub type ZarrArray;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn name(this: &ZarrArray) -> Option<String>;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn basename(this: &ZarrArray) -> Option<String>;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn shape(this: &ZarrArray) -> Vec<u32>;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn chunks(this: &ZarrArray) -> Vec<u32>;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "chunkSize")]
    pub fn chunk_size(this: &ZarrArray) -> u32;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "dtype")]
    pub fn data_type_js(this: &ZarrArray) -> JsValue;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "fillValue")]
    pub fn fill_value(this: &ZarrArray) -> f32;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "nDims")]
    pub fn number_of_dimensions(this: &ZarrArray) -> u32;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn size(this: &ZarrArray) -> u32;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore")]
    pub fn length(this: &ZarrArray) -> u32;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "chunkDataShape")]
    pub fn chunk_data_shape(this: &ZarrArray) -> Vec<u32>;

    #[wasm_bindgen(method, getter, js_class = "ZarrStore", js_name = "numChunks")]
    pub fn number_of_chunks(this: &ZarrArray) -> u32;

    #[wasm_bindgen(method, js_class = "ZarrArray", js_name = "getRaw")]
    pub async fn get_raw(this: &ZarrArray, selection: JsValue, options: JsValue) -> JsValue;

    #[wasm_bindgen(js_name = "openArray")]
    async fn open_array(options: JsValue) -> JsValue;
}

impl ZarrArray {
    pub async fn open(store: String, path: String) -> ZarrArray {
        let options = OpenArrayOptions {
            store,
            path: Some(path),
            ..Default::default()
        };
        let array = open_array(serde_wasm_bindgen::to_value(&options).unwrap()).await;
        array.unchecked_into()
    }

    pub async fn get_raw_data(
        &self,
        selection: Option<Vec<DimensionArraySelection>>,
        options: GetOptions,
    ) -> RawArray {
        self.get_raw(
            serde_wasm_bindgen::to_value(&selection).unwrap(),
            serde_wasm_bindgen::to_value(&options).unwrap(),
        )
        .await
        .unchecked_into::<RawArray>()
    }

    pub fn data_type(&self) -> DataType {
        serde_wasm_bindgen::from_value(self.data_type_js()).unwrap()
    }
}
