@include(octree_node)
@include(octree_node_util)

fn node_idx_store_global(node_index: u32, node: u32) {
    octree_nodes[node_index] = node;
}

fn node_idx_store(node_index: u32, node: u32, num_channels: u32, channel_index: u32) -> u32 {
    node_idx_store_global(to_multichannel_node_index(node_index, num_channels, channel_index), node);
}

fn node_idx_set_min_max_global(node_index: u32, new_min: u32, new_max: u32) {
    var node = node_idx_load_global(node_index);
    node_set_min(&node, new_min);
    node_set_max(&node, new_max);
    node_idx_store_global(node_index, node);
}

fn node_idx_set_min_max(node_index: u32, new_min: u32, new_max: u32, num_channels: u32, channel_index: u32) {
    node_idx_set_min_max_global(to_multichannel_node_index(node_index, num_channels, channel_index), new_min, new_max);
}

fn node_idx_set_partially_mapped_resolutions_global(node_index: u32, new_partially_mapped_resolutions_bitmask: u32) {
    var node = node_idx_load_global(node_index);
    node_set_partially_mapped_resolutions(&node, new_partially_mapped_resolutions_bitmask);
    node_idx_store_global(node_index, node);
}

fn node_idx_set_partially_mapped_resolutions(node_index: u32, mapped_resolutions_bitmask: u32, num_channels: u32, channel_index: u32) {
    node_idx_set_partially_mapped_resolutions_global(to_multichannel_node_index(node_index, num_channels, channel_index), mapped_resolutions_bitmask);
}
