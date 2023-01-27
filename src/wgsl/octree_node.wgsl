// NODE LAYOUT
// ----------------------------------------------------------------------------
// Each octree ndoe is an unsigned 32-bit integer.

const NODE_MIN_OFFSET: u32 = 0;
const NODE_MIN_COUNT: u32 = 8;
const NODE_MAX_OFFSET: u32 = NODE_MIN_OFFSET + NODE_MIN_COUNT;
const NODE_MAX_COUNT: u32 = NODE_MIN_COUNT;
const MAPPED_RESOLUTIONS_OFFSET: u32 = NODE_MAX_OFFSET + NODE_MAX_COUNT;
const MAPPED_RESOLUTIONS_COUNT: u32 = (32 - NODE_MIN_COUNT - NODE_MAX_COUNT) / 2;
const PARTIALLY_MAPPED_RESOLUTIONS_OFFSET = MAPPED_RESOLUTIONS_OFFSET + MAPPED_RESOLUTIONS_COUNT;
const PARTIALLY_MAPPED_RESOLUTIONS_COUNT: u32 = MAPPED_RESOLUTIONS_COUNT;

// NODE OPERATIONS
// ----------------------------------------------------------------------------

fn node_new(min_value: u32, max_value: u32, mapped_bitmask: u32, partially_mapped_bitmask: u32) -> u32 {
    var node = 0;
    node = insertBits(node, min_value, NODE_MIN_OFFSET, NODE_MIN_COUNT);
    node = insertBits(node, max_value, NODE_MAX_OFFSET, NODE_MAX_COUNT);
    node = insertBits(node, mapped_bitmask, MAPPED_RESOLUTIONS_OFFSET, MAPPED_RESOLUTIONS_COUNT);
    node = insertBits(node, partially_mapped_bitmask, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
    return node;
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

fn node_get_mapped_resolutions(node: u32) -> u32 {
    return extractBits(node, MAPPED_RESOLUTIONS_OFFSET, MAPPED_RESOLUTIONS_COUNT);
}

fn node_set_mapped_resolutions(node: ptr<function, u32>, new_mapped_resolutions_bitmask: u32) {
    *node = insertBits(*node, new_mapped_resolutions_bitmask, MAPPED_RESOLUTIONS_OFFSET, MAPPED_RESOLUTIONS_COUNT);
}

fn node_get_partially_mapped_resolutions(node: u32) -> u32 {
    return extractBits(node, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
}

fn node_set_partially_mapped_resolutions(node: ptr<function, u32>, new_partially_mapped_resolutions_bitmask: u32) {
    *node = insertBits(*node, new_partially_mapped_resolutions_bitmask, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT);
}

fn node_make_mask_for_resolution(resolution: u32) -> u32 {
    return insertBits(0, 1, resolution, 1);
}
