@group(0)
@binding(0)
var inputImage: texture_3d<u32>;

@group(0)
@binding(1)
var resultImage: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    z_slice: i32,
    z_max: f32
}

@group(0)
@binding(2)
var<uniform> z_slice: Uniforms;

@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let window_size = textureDimensions(result);
    if window_size.x < global_id.x || window_size.y < global_id.y {
        return;
    }

    let pixel = vec2<i32>(global_id.xy);

    let raw_value = textureLoad(inputImage, vec3<i32>(pixel, z_slice.z_slice), 0).x;
    let value = vec3<f32>(f32(raw_value) / z_slice.z_max);
    textureStore(resultImage, pixel, vec4<f32>(value, 1.0));
}
