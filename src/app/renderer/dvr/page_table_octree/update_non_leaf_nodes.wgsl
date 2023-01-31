@include(multichannel_octree_util)
@include(octree_node)
@include(volume_subdivision_util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<uniform> page_directory_meta: PageDirectoryMeta;

// (read-only) cache update bind group
@group(1) @binding(0) var<uniform> subdivision_index: u32;
// node indices, reset to 255
@group(1) @binding(1) var<storage, read_write> node_helper_buffer_a: array<u32>;

// (read-write) output nodes
@group(2) @binding(0) var<storage, read_write> octree_nodes: array<u32>;
// parent node indices
@group(2) @binding(2) var<storage, read_write> node_helper_buffer_b: array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn update_non_leaf_nodes(@builtin(global_invocation_id) global_invocation_id) {
    let num_channels = page_directory_meta.max_channels;

    let global_id = global_invocation_id.x;
    let num_nodes = subdivision_idx_num_nodes_in_subdivision(subdivision_index) * num_channels;
    if (global_id >= num_nodes) {
        return;
    }

    let multichannel_local_node_index = node_helper_buffer_a[global_id];
    node_helper_buffer_a[global_id] = 255;

    let offset = subdivision_idx_get_node_offset(subdivision_index) * num_channels;
    let global_node_index = offset + multichannel_local_node_index;
    let single_channel_local_index = multi_local_to_single_channel_local_index(multichannel_local_node_index, num_channels);

    let num_child_nodes = subdivision_idx_get_children_per_node(subdivision_index);
    var child_index = to_multichannel_node_index(
        subdivision_idx_first_child_index_from_local(subdivision_index, single_channel_local_index),
        num_channels,
        channel_index
    );

    var minimum = 255;
    var maximum = 0;
    var partially_mapped_resolutions = 0;
    for (var i = 0; i < num_child_nodes; i += 1) {
        let child_node = node_idx_load_global(child_index);

        minimum = min(minimum, node_get_min(child_node));
        maximum = max(maximum, node_get_max(child_node));
        partially_mapped_resolutions |= node_get_partially_mapped_resolutions(child_node);

        child_index += num_channels;
    }

    let old_node = octree_nodes[global_node_index];
    let new_node = node_new(minimum, maximum, partially_mapped_resolutions);
    let changed = old_node != new_node;
    if (changed) {
        octree_nodes[global_node_index] = new_node;
        if (subdivision_index > 0) {
            let parent_node_index = to_multichannel_node_index(
                subdivision_idx_local_parent_node_index_from_local(subdivision_index, single_channel_local_index),
                num_channels,
                channel_index
            );
            node_helper_buffer_b[parent_node_index] = u32(true);
        }
    }
}