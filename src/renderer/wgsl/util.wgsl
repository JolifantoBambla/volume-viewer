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
