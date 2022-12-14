// This interface requires the following identifiers to be defined:
// bindings:
//  -
//
// functions (implementation of the interface):
//  - _volume_acessor__compute_lod(distance: f32, channel: u32, lod_factor: f32) -> u32
//  - _volume_accessor__get_node(lod: u32, channel: u32, request_bricks: bool) -> Node

@include(type_alias)

struct Node {
    has_average: bool,
    average: f32,
    is_mapped: bool,
    sample_address: float3,
    requested_brick: bool,
}

fn va_compute_lod(distance: f32, channel: u32, lod_factor: f32) -> u32 {
    return _volume_acessor__compute_lod(distance, channel, lod_factor);
}

fn va_get_node(ray_sample: float3, lod: u32, channel: u32, request_bricks: bool) -> Node {
    return _volume_accessor__get_node(ray_sample, lod, channel, request_bricks);
}
