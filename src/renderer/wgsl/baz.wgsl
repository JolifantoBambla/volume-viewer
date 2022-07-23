@include(type_alias.wgsl)

fn x(y: f32) -> { return y; }

@include(bar.wgsl)
@include(foo.wgsl)

fn a(b: f32) -> { return b; }
