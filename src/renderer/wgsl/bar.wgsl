@include(type_alias.wgsl)

fn bar(b: float3) -> f32 {
    return b.r;
}
