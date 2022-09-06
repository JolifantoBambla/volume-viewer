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
    // todo: make const
    /*
    let comparison_to_axis = array<u32, 8>(2u, 1u, 2u, 1u, 2u, 2u, 0u, 0u);
    let comparison = (u32(v[0] < v[1]) << 2u) + (u32(v[0] < v[2]) << 1u) + u32(v[1] < v[2]);
    return comparison_to_axis[comparison];
    */

    if (v.x < v.y && v.x < v.z) {
        return 0u;
    } else if (v.y < v.z) {
        return 1u;
    } else {
        return 2u;
    }
}

fn clamp_to_one(v: float3) -> float3 {
    return clamp(v, float3(), float3(1.));
}