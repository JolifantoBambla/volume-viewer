@include(multichannel_octree_util)
@include(octree_node)
@include(octree_node_util)
@include(page_table)
@include(volume_subdivision)
@include(volume_subdivision_util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<uniform> page_directory_meta: PageDirectoryMeta;

// (read-only) cache update bind group
@group(1) @binding(0) var<uniform> subdivision_index: u32;
@group(1) @binding(1) var<storage> num_nodes_to_update: u32;
// node indices, reset to 255
@group(1) @binding(2) var<storage, read_write> node_helper_buffer_a: array<u32>;

// (read-write) output nodes
@group(2) @binding(0) var<storage, read_write> octree_nodes: array<u32>;
// parent node indices
@group(2) @binding(1) var<storage, read_write> node_helper_buffer_b: array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let num_channels = page_directory_meta.max_channels;

    let global_id = global_invocation_id.x;
    if (global_id >= num_nodes_to_update) {
        return;
    }

    let multichannel_local_node_index = node_helper_buffer_a[global_id];
    node_helper_buffer_a[global_id] = 255;

    let offset = subdivision_idx_get_node_offset(subdivision_index) * num_channels;
    let global_node_index = offset + multichannel_local_node_index;
    let single_channel_local_index = multi_local_to_single_channel_local_index(multichannel_local_node_index, num_channels);
    let channel_index = multi_local_to_channel_index(multichannel_local_node_index, num_channels);

    let num_child_nodes = subdivision_idx_get_children_per_node(subdivision_index);
    var child_index = to_multichannel_node_index(
        subdivision_idx_first_child_index_from_local(subdivision_index, single_channel_local_index),
        num_channels,
        channel_index
    );

    var minimum: u32 = 255;
    var maximum: u32 = 0;
    var partially_mapped_resolutions: u32 = 0;

    let single_channel_local_subscript = index_to_subscript(
        single_channel_local_index,
        subdivision_idx_get_shape(subdivision_index)
    );
    let single_channel_first_child_subscript = single_channel_local_subscript * subdivision_idx_get_node_shape(subdivision_index);

    /*
    index_to_subscript(
        // todo: maybe this is wrong?
        // todo: if so, computing the parent index might also be wrong?
        // use subscript * node_shape instead?
        subdivision_idx_first_child_index_from_local(subdivision_index, single_channel_local_index),
        subdivision_idx_get_shape(subdivision_index + 1)
    );
    */
    // todo: check if correct!
    let node_shape = subdivision_idx_get_node_shape(subdivision_index);
    for (var x: u32 = 0; x < node_shape.x; x += 1) {
        for (var y: u32 = 0; y < node_shape.y; y += 1) {
            for (var z: u32 = 0; z < node_shape.z; z += 1) {
                let child_node_index = to_multichannel_node_index(
                    subdivision_idx_global_node_index(
                        subdivision_index,
                        subdivision_idx_subscript_to_local_index(
                            subdivision_index,
                            vec3<u32>(single_channel_first_child_subscript + vec3(x, y, z))
                        )
                    ),
                    num_channels,
                    channel_index
                );

                let child_node = node_idx_load_global(child_node_index);
                minimum = min(minimum, node_get_min(child_node));
                maximum = max(maximum, node_get_max(child_node));
                partially_mapped_resolutions |= node_get_partially_mapped_resolutions(child_node);
            }
        }
    }

    /*
    for (var i: u32 = 0; i < num_child_nodes; i += 1) {
        let child_node = node_idx_load_global(child_index);

        minimum = min(minimum, node_get_min(child_node));
        maximum = max(maximum, node_get_max(child_node));
        partially_mapped_resolutions |= node_get_partially_mapped_resolutions(child_node);

        child_index += num_channels;
    }
    */

    let old_node = octree_nodes[global_node_index];
    let new_node = node_new(minimum, maximum, partially_mapped_resolutions);
    let changed = old_node != new_node;
    if (changed) {
        octree_nodes[global_node_index] = new_node;
        if (subdivision_index > 0) {
            // todo: check parent node index!
            let single_channel_parent_node_subscript = single_channel_local_subscript / subdivision_idx_get_node_shape(subdivision_index - 1);
            let parent_node_index =  to_multichannel_node_index(
                subdivision_idx_subscript_to_local_index(subdivision_index - 1, single_channel_parent_node_subscript),
                num_channels,
                channel_index
            );

            /*
            to_multichannel_node_index(
                subdivision_idx_local_parent_node_index_from_local(subdivision_index, single_channel_local_index),
                num_channels,
                channel_index
            );
            */

            node_helper_buffer_b[parent_node_index] = u32(true);
        }
    }
}