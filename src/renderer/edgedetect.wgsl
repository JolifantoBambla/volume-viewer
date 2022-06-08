@group(0)
@binding(0)
var inputImage: texture_2d<f32>;

@group(0)
@binding(1)
var resultImage: texture_storage_2d<rgba8unorm, write>;

fn conv(kernel: ptr<function, array<f32, 9u>>, data: ptr<function, array<f32, 9u>>, denom: f32, offset: f32) -> f32 {
    var res: f32 = 0.0;
    for (var i: i32 = 0; i < 9; i = i + 1) {
        res = (res + ((*kernel)[i] * (*data)[i]));
    }
    return clamp(((res / (denom)) + (offset)), 0.0, 1.0);
}

@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = bitcast<vec2<i32>>(global_id.xy);
    var average: array<f32,9u>;

    var n: i32 = -1;
    for (var i: i32 = -1; i < 2; i = i + 1) {
        for (var j: i32 = -1; j < 2; j = j + 1) {
            n = n + 1;
            let rgb: vec3<f32> = textureLoad(inputImage, vec2<i32>(pixel.x + i, pixel.y + j), 0).xyz;
            average[n] = (rgb.r + rgb.g + rgb.b) / 3.0;
        }
    }

    var kernel = array<f32,9u>(
      -0.125, -0.125, -0.125,
      -0.125,  1.0,   -0.125,
      -0.125, -0.125, -0.125
    );

    let res = vec4<f32>(vec3<f32>(conv(&kernel, &average, 0.1, 0.0)), 1.0);
    textureStore(resultImage, pixel, res);
}
