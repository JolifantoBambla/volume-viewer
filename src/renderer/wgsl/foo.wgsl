@include(type_alias.wgsl)

fn foo(f: float4) -> f32 {
    return f.r;
}
