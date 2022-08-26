@include(type_alias)

// constant values
// todo: replace let with const
// no idea how infinity is written out in WGSL, so here are some constants
// note: no compile-time expressions without const -> needs to be a function...
fn positive_infinity() -> f32 { return  1. / 0.; }
fn negative_infinity() -> f32 { return -1. / 0.; }

let EPSILON = 0.000001;

let RED = float4(1., 0., 0., 1.);
let GREEN = float4(0., 1., 0., 1.);
let BLUE = float4(0., 0., 1., 1.);
let WHITE = float4(1.);
let BLACK = float4(0., 0., 0., 1.);
