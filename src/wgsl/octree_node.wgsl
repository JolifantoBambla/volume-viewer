// NODE LAYOUT
// ----------------------------------------------------------------------------
// Each octree ndoe is an unsigned 32-bit integer.

const NODE_MIN_OFFSET: u32 = 0;
const NODE_MIN_COUNT: u32 = 8;
const NODE_MAX_OFFSET: u32 = NODE_MIN_OFFSET + NODE_MIN_COUNT;
const NODE_MAX_COUNT: u32 = NODE_MIN_COUNT;
const PARTIALLY_MAPPED_RESOLUTIONS_OFFSET: u32 = NODE_MAX_OFFSET + NODE_MAX_COUNT;
const PARTIALLY_MAPPED_RESOLUTIONS_COUNT: u32 = (32 - NODE_MIN_COUNT - NODE_MAX_COUNT);

// NODE OPERATIONS
// ----------------------------------------------------------------------------

fn node_new(min_value: u32, max_value: u32, partially_mapped_bitmask: u32) -> u32 {
    var node: u32 = 0;
    node = insertBits(node, min_value, NODE_MIN_OFFSET, NODE_MIN_COUNT);
    node = insertBits(node, max_value, NODE_MAX_OFFSET, NODE_MAX_COUNT);
    node = insertBits(node, partially_mapped_bitmask, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
    return node;
}

fn node_has_no_data(node: u32) -> bool {
    // todo: enable other line, this is just for debugging
    return node == 255;
    //return node_get_min(node) == 255 && node_get_max(node) == 0 && node_get_partially_mapped_resolutions(node) == 0;
//    return node_get_min(node) > node_get_max(node);
}

fn node_is_not_mapped(node: u32) -> bool {
    return node_get_partially_mapped_resolutions(node) == 0;
}

fn node_is_empty(node: u32, minimum: u32, maximum: u32) -> bool {
    return (minimum > node_get_max(node)) || (maximum < node_get_min(node));
}

/*
todo: @include(constant)
fn node_is_empty_f32(node: u32, lower_threshold: f32, upper_threshold: f32) -> bool {
    let lower_bound = u32(floor((lower_threshold - EPSILON) * 255.0));
    let upper_bound = u32(floor((upper_threshold + EPSILON) * 255.0));
    return node_is_empty(node, lower_bound, upper_bound);
}
*/

fn node_is_homogeneous(node: u32, threshold: f32) -> bool {
    return distance(f32(node_get_max(node)), f32(node_get_min(node))) < threshold;
}

fn node_get_min(node: u32) -> u32 {
    return extractBits(node, NODE_MIN_OFFSET, NODE_MIN_COUNT);
}

fn node_set_min(node: ptr<function, u32>, new_min: u32) {
    *node = insertBits(*node, new_min, NODE_MIN_OFFSET, NODE_MIN_COUNT);
}

fn node_get_max(node: u32) -> u32 {
    return extractBits(node, NODE_MAX_OFFSET, NODE_MAX_COUNT);
}

fn node_set_max(node: ptr<function, u32>, new_max: u32) {
    *node = insertBits(*node, new_max, NODE_MAX_OFFSET, NODE_MAX_COUNT);
}

fn node_get_partially_mapped_resolutions(node: u32) -> u32 {
    return extractBits(node, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
}

fn node_set_partially_mapped_resolutions(node: ptr<function, u32>, new_partially_mapped_resolutions_bitmask: u32) {
    *node = insertBits(*node, new_partially_mapped_resolutions_bitmask, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
}

fn node_make_mask_for_resolution(resolution: u32) -> u32 {
    return insertBits(0u, 1, resolution, 1);
}

fn node_resolution_is_partially_mapped(node: u32, resolution: u32) -> bool {
    return (node_get_partially_mapped_resolutions(node) & node_make_mask_for_resolution(resolution)) > 0;
}
