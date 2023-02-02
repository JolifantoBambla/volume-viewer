@include(dispatch_indirect)
@include(multichannel_octree_util)
@include(octree_node)
@include(page_table)
@include(volume_subdivision)
@include(volume_subdivision_util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<uniform> page_directory_meta: PageDirectoryMeta;

// (read-only) cache update bind group
// resolution mapping & min max changed
@group(1) @binding(0) var<storage, read_write> node_helper_buffer_b: array<u32>;

// (read-write) output nodes
@group(2) @binding(0) var<storage, read_write> octree_nodes: array<u32>;
@group(2) @binding(1) var<storage, read_write> next_level_update_indirect: DispatchWorkgroupsIndirect;
@group(2) @binding(2) var<storage, read_write> num_nodes_next_level: atomic<u32>;
// parent node indices
@group(2) @binding(3) var<storage, read_write> node_helper_buffer_a: array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let subdivision_index = subdivision_get_leaf_node_level_index();
    let num_channels = page_directory_meta.max_channels;

    let multichannel_local_node_index = global_invocation_id.x;
    let num_nodes = subdivision_idx_num_nodes_in_subdivision(subdivision_index) * num_channels;
    if (multichannel_local_node_index >= num_nodes) {
        return;
    }

    let node_update = node_helper_buffer_b[multichannel_local_node_index];

    // clean up for next passes
    node_helper_buffer_b[multichannel_local_node_index] = 0;

    if (node_update > 0) {
        // update node
        let offset = subdivision_idx_get_node_offset(subdivision_index) * num_channels;
        let global_node_index = offset + multichannel_local_node_index;
        var node = octree_nodes[global_node_index];

        let resolution_mapping = node_get_partially_mapped_resolutions(node_update);
        let updated_resolution_mapping = node_get_partially_mapped_resolutions(node) | resolution_mapping;
        node_set_partially_mapped_resolutions(&node, updated_resolution_mapping);

        octree_nodes[global_node_index] = node;

        let channel_index = multi_local_to_channel_index(multichannel_local_node_index, num_channels);
        let multichannel_local_parent_node_index = to_multichannel_node_index(
            subdivision_idx_local_parent_node_index(
                subdivision_index,
                multi_local_to_single_channel_local_index(multichannel_local_node_index, num_channels),
            ),
            num_channels,
            channel_index
        );
        let index = atomicAdd(&num_nodes_next_level, 1);
        atomicMax(&next_level_update_indirect.workgroup_count_x, max(index / 64, 1));
        node_helper_buffer_a[index] = multichannel_local_parent_node_index;
    }
}