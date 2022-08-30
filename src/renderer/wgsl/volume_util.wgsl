/// Requires the following functions to be defined in the shader:
///  - sample_volume(vec3<f32>) -> f32

@include(type_alias)

fn central_differences(x: float3, step: float3) -> float3 {
    var central_differences = float3();
    for (var i = 0; i < 3; i += 1) {
        let h = step[i];
        var offset = float3();
        offset[i] = h;
        central_differences[i] = sample_volume(x - offset) - sample_volume(x + offset) / (2. * h);
    }
    return central_differences;
}

fn max_in_neighborhood(x: float3, step: float3) -> f32 {
    var max_value = 0.;
    for (var i = 0; i < 3; i += 1) {
        let h = step[i];
        var offset = float3();
        offset[i] = h;
        max_value = max(max_value, max(sample_volume(x - offset), sample_volume(x + offset)));
    }
    return max_value;
}

/// Computes the normal at point x from the central difference.
fn compute_volume_normal(x: float3, step: float3, view: float3) -> float3 {
    let normal = normalize(central_differences(x, step));
    // flip normal towards viewer if necessary
    let front_or_back = 2. * f32(dot(normal, view) > 0.) - 1.;
    return front_or_back * normal;
}
