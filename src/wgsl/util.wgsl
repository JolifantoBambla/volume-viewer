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

fn unpack4x8uint(e: u32) -> vec4<u32> {
    return vec4<u32>(
        extractBits(e, 24, 8),
        extractBits(e, 16, 8),
        extractBits(e, 8, 8),
        extractBits(e, 0, 8)
    );
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

fn subscript_to_normalized_address(subscript: uint3, extent: uint3) -> float3 {
    return float3(subscript) / float3(extent);
}

fn normalized_address_to_subscript(normalized_address: float3, extent: uint3) -> uint3 {
    // todo: shouldn't this be floor((extent - 1) * normalized_address)?
    return min(uint3(floor(float3(extent) * normalized_address)), extent - uint3(1u));
}

fn index_to_subscript(index: u32, extent: uint3) -> uint3 {
    let x = index % extent.x;
    let y = ((index - x) / extent.x) % extent.y;
    let z = (((index - x) / extent.x) - y) / extent.y;
    return uint3(x, y, z);
}

fn subscript_to_index(subscript: uint3, extent: uint3) -> u32 {
    return subscript.x + extent.x * (subscript.y + extent.y * subscript.z);
}

fn clamp_to_one(v: float3) -> float3 {
    return clamp(v, float3(), float3(1.));
}

// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
fn wang_hash(seed: u32) -> u32 {
	var h = (seed ^ 61) ^ (seed >> 16);
	h *= 9;
	h = h ^ (h >> 4);
	h *= 0x27d4eb2d;
	return h ^ (h >> 15);
}

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
fn pcg_hash(seed: u32) -> u32 {
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_u32_to_f32(h: u32) -> f32 {
    return f32(h % 2147483647) / f32(2147483647);
}

/// Computes a level of detail within a given highest and lowest level for the distance from the current sample to the
/// camera in the camera's space.
/// The ranges of the given highest and lowest levels are max(0, highest_res) and max(highest_res, lowest_res)
/// respectively.
/// It is the caller's responsibility to choose an adequate factor.
fn select_level_of_detail(distance: f32, highest_res: u32, lowest_res: u32, lod_factor: f32) -> u32 {
    let lod = u32(log2(lod_factor * distance));
    return clamp(lod, highest_res, lowest_res);
}
