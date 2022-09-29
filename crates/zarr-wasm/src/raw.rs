use wasm_bindgen::prelude::*;

use crate::data_type::DataType;

#[wasm_bindgen(module = "https://cdn.skypack.dev/zarr")]
extern "C" {
    #[wasm_bindgen(js_name = "RawArray", typescript_type = "RawArray")]
    #[derive(Debug, Clone)]
    pub type RawArray;

    #[wasm_bindgen(method, getter, js_class = "RawArray", js_name = "data")]
    pub fn data_uint16(this: &RawArray) -> Vec<u16>;

    //  todo: other data types (macro)

    #[wasm_bindgen(method, getter, js_class = "RawArray", js_name = "dtype")]
    pub fn data_type_js(this: &RawArray) -> String;

    #[wasm_bindgen(method, getter, js_class = "RawArray")]
    pub fn shape(this: &RawArray) -> Vec<u32>;

    #[wasm_bindgen(method, getter, js_class = "RawArray")]
    pub fn strides(this: &RawArray) -> Vec<u32>;
    /*
    data: Uint16Array(2621440) [1, 6, 10, 14, 16, 20, 23, 26, 27, 26, 24, 19, 11, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 16, 18, 20, 22, 22, 24, 28, 32, 34, 35, 36, 38, 36, 37, 38, 42, 42, 42, 41, 40, 34, 30, 27, 26, 26, 28, 33, 39, 46, 47, 49, 50, 49, 47, 48, 48, 46, 40, 31, 19, 10, 0, 0, 0, 0, 0, 0, 0, 0, 23, 34, 40, 44, 46, 40, 28, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …]
    dtype: ">u2"
    shape: (3) [40, 256, 256]
    strides: (3) [65536, 256, 1]
     */
}

// todo: get typed array from raw array
impl RawArray {
    pub fn get_data_type(&self) -> DataType {
        serde_json::from_str(&self.data_type_js()).unwrap()
    }

    pub fn get_data_uint16(&self) -> Vec<u16> {
        let data_type = self.get_data_type();
        match data_type {
            DataType::Uint16LittleEndian => {
                #[cfg(target_endian = "little")]
                {
                    self.data_uint16()
                }
                #[cfg(not(target_endian = "little"))]
                {
                    self.data_uint16()
                        .iter()
                        .map(|x| u16::from_le(*x))
                        .collect()
                }
            }
            DataType::Uint16BigEndian => {
                #[cfg(target_endian = "little")]
                {
                    self.data_uint16()
                        .iter()
                        .map(|x| u16::from_be(*x))
                        .collect()
                }
                #[cfg(not(target_endian = "little"))]
                {
                    self.data_uint16()
                }
            }
            _ => panic!("Raw array is not a u16 array"),
        }
    }
}
