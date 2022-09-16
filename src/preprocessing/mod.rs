use rayon::prelude::*;
use wasm_bindgen::prelude::*;
pub use wasm_bindgen_rayon::init_thread_pool;

// todo: byte order conversions

// IS ZERO

#[wasm_bindgen(js_name = "isZeroU8")]
pub fn is_zero_u8(data: Vec<u8>, threshold: f32) -> bool {
    let thresh = (threshold.clamp(0., 1.) * 255.) as u8;
    data.par_iter().all(|&x| x <= thresh)
}

#[wasm_bindgen(js_name = "isZeroU16")]
pub fn is_zero_u16(data: Vec<u16>, threshold: f32) -> bool {
    let thresh = (threshold.clamp(0., 1.) * 65535.) as u16;
    data.par_iter().all(|&x| x <= thresh)
}

#[wasm_bindgen(js_name = "isZeroF32")]
pub fn is_zero_f32(data: Vec<f32>, threshold: f32) -> bool {
    let thresh = threshold.clamp(0., 1.);
    data.par_iter().all(|&x| x.abs() <= thresh)
}

// MIN

#[wasm_bindgen(js_name = "minU8")]
pub fn min_u8(data: Vec<u8>) -> f32 {
    *data.par_iter().min().unwrap() as f32
}

#[wasm_bindgen(js_name = "minU16")]
pub fn min_u16(data: Vec<u16>) -> f32 {
    *data.par_iter().min().unwrap() as f32
}

#[wasm_bindgen(js_name = "minF32")]
pub fn min_f32(data: Vec<f32>) -> f32 {
    *data.par_iter().min_by(|&a, &b| a.partial_cmp(b).expect("Encountered NaN value in data")).unwrap()
}

// MAX

#[wasm_bindgen(js_name = "maxU8")]
pub fn max_u8(data: Vec<u8>) -> f32 {
    *data.par_iter().max().unwrap() as f32
}

#[wasm_bindgen(js_name = "maxU16")]
pub fn max_u16(data: Vec<u16>) -> f32 {
    *data.par_iter().max().unwrap() as f32
}

#[wasm_bindgen(js_name = "maxF32")]
pub fn max_f32(data: Vec<f32>) -> f32 {
    *data.par_iter().max_by(|&a, &b| a.partial_cmp(b).expect("Encountered NaN value in data")).unwrap()
}

// SCALE

#[wasm_bindgen(js_name = "scaleU8ToU8")]
pub fn scale_u8_to_u8(data: Vec<u8>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| ((x as f32 / max_value) * 255.) as u8)
        .collect()
}

#[wasm_bindgen(js_name = "scaleU16ToU8")]
pub fn scale_u16_to_u8(data: Vec<u16>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| ((x as f32 / max_value) * 255.) as u8)
        .collect()
}

#[wasm_bindgen(js_name = "scaleF32ToU8")]
pub fn scale_f32_to_u8(data: Vec<f32>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| ((x / max_value) * 255.) as u8)
        .collect()
}

// LOG

#[wasm_bindgen(js_name = "logTransformU8")]
pub fn log_transform_u8(data: Vec<u8>) -> Vec<u8> {
    data.par_iter()
        .map(|&x| (f32::log10(x as f32 / 255.) * 255.) as u8)
        .collect()
}

#[wasm_bindgen(js_name = "logTransformU16")]
pub fn log_transform_u16(data: Vec<u16>) -> Vec<u16> {
    data.par_iter()
        .map(|&x| (f32::log10(x as f32 / 65535.) * 65535.) as u16)
        .collect()
}

#[wasm_bindgen(js_name = "logTransformF32")]
pub fn log_transform_f32(data: Vec<f32>) -> Vec<f32> {
    data.par_iter()
        .map(|&x| f32::log10(x))
        .collect()
}

// LOG & SCALE

#[wasm_bindgen(js_name = "logScaleU8")]
pub fn log_scale_u8(data: Vec<u8>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| (f32::log10(x as f32 / max_value) * 255.) as u8)
        .collect()
}

#[wasm_bindgen(js_name = "logScaleU16")]
pub fn log_scale_u16(data: Vec<u16>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| (f32::log10(x as f32 / max_value) * 255.) as u8)
        .collect()
}

#[wasm_bindgen(js_name = "logScaleF32")]
pub fn log_scale_f32(data: Vec<f32>, max_value: f32) -> Vec<u8> {
    data.par_iter()
        .map(|&x| (f32::log10(x / max_value) * 255.) as u8)
        .collect()
}
