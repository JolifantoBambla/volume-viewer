@include(type_alias)

fn swap(a: ptr<function, f32, read_write>, b: ptr<function, f32, read_write>) {
    let helper = *a;
    *a = *b;
    *b = helper;
}

fn blend(a: float4, b: float4) -> float4 {
    return float4(
        a.rgb * a.a + b.rgb * b.a * (1. - a.a),
        a.a + b.a * (1. - a.a)
    );
}

fn sum_components_uint4(v: uint4) -> u32 {
    return v.x + v.y + v.z + v.w;
}

fn pack4x8uint(e: uint4) -> u32 {
    return sum_components_uint4(e << uint4(24u, 16u, 8u, 0u));
}

fn min_dimension(v: float3) -> u32 {
    let comparison_to_axis = array<u32, 8>(2u, 1u, 2u, 1u, 2u, 2u, 0u, 0u);
    let comparison = (u32(v[0] < v[1]) << 2u) + (u32(v[0] < v[2]) << 1u) + u32(v[1] < v[2]);
    return comparison_to_axis[comparison];
}

fn max_dimension(v: float3) -> u32 {
    let comparison_to_axis = array<u32, 8>(2u, 1u, 2u, 1u, 2u, 2u, 0u, 0u);
    let comparison = (u32(v[0] > v[1]) << 2u) + (u32(v[0] > v[2]) << 1u) + u32(v[1] > v[2]);
    return comparison_to_axis[comparison];
}

fn min_component(v: float3) -> f32 {
    return v[min_dimension(v)];
}

fn max_component(v: float3) -> f32 {
    return v[max_dimension(v)];
}

fn clamp_to_one(v: float3) -> float3 {
    return clamp(v, float3(), float3(1.));
}

// Pseudo-random number gen from
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// with some tweaks for the range of values
// https://github.com/johanna-b/VisWeb/blob/2d414949cd1911cdd55a1ef7d0b5b385ede892c2/shaders/vol-shader.js#L55
fn wang_hash(seed: i32) -> f32 {
	var h = (seed ^ 61) ^ (seed >> 16);
	h *= 9;
	h = h ^ (h >> 4);
	h *= 0x27d4eb2d;
	h = h ^ (h >> 15);
	return f32(h % 2147483647) / f32(2147483647);
}