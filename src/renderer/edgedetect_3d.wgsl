@group(0)
@binding(0)
var inputImage: texture_3d<f32>;

@group(0)
@binding(1)
var resultImage: texture_storage_2d<rgba8unorm, write>;

@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = bitcast<vec2<i32>>(global_id.xy);

    let value = textureLoad(inputImage, vec3<i32>(pixel, 0), 0).x;
    if (value > 0.5) {
        textureStore(resultImage, pixel, vec4<f32>(value, 0.0, 0.0, 1.0));
    } else {
    textureStore(resultImage, pixel, vec4<f32>(0.0,0.0,0.0,1.0));
    }
}
