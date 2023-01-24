@include(dispatch_indirect)
@include(octree_node)
@include(octree_node_util)
@include(page_table)
@include(volume_subdivision)
@include(volume_subdivision_util)
@include(util)

@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<storage, read_write> octree_nodes: array<u32>;

// MULTICHANNEL INDEX OPERATIONS
// ----------------------------------------------------------------------------

// Computes the global node index in a list of interleaved octree nodes from a local one, i.e., an index that is unaware
// of other channels in the same octree
fn to_multichannel_node_index(local_node_index: u32, num_channels: u32, channel_index: u32) -> u32 {
    return local_node_index * num_channels + channel_index;
}

fn to_channel_local_node_index(multichannel_node_index: u32, num_channels: u32) -> u32 {
}

struct OctreeLevelUpdateMeta {
    num_update_nodes: atomic<u32>,
    subidivision_index: u32,
}

@group(0) @binding(0) var<storage, read_write> num_update_nodes: atomic<u32>;
@group(0) @binding(0) var<storage, read_write> update_node_indices: array<u32>;
@group(0) @binding(0) var<storage, read_write> next_level_update_node_indices: array<u32>;
@group(0) @binding(0) var<storage, read_write> next_level_update_indirect: DispatchWorkgroupsIndirect;

/// The whole process looks like this:
/// - map mapped & unmapped bricks to affected leaf nodes
/// - update node min & max values
/// -
/// It starts with brick indices being mapped to node indices


@group(0) @binding(0) var<storage> num_unmapped_brick_ids: u32;
@group(0) @binding(0) var<storage> num_mapped_brick_ids: u32;
@group(0) @binding(0) var<storage> unmapped_brick_ids: array<u32>;
@group(0) @binding(0) var<storage> mapped_brick_ids: array<u32>;

@group(0) @binding(0) var<uniform> page_directory_meta: PageDirectoryMeta;

// these two buffers have size num_nodes_in_highest_subdivision * num_channels
// they are used for multiple things:
// 1st pass: compute minima & maxima:
//  - a) minima of nodes
//  - b) maxima of nodes
// 2nd pass: update minima & maxima:
//  - write minima of a) and maxima of b) to nodes
// 3rd pass: process brick updates
//  - a) mark node indices that have been updated with a non-zero bitmask of resolutions that have been changed (atomicOr)
// 4th pass: update mapped states
//  - use masks in a) to update nodes if mask is non-zero (just use 'or' because one thread per node)
//  - b) if changed, mark parent indices
// 5th pass: pack parent node indices
//  - a) store marked indices in b)
// 6th pass: update nodes on next level
//  - b) mark parent indices
// repeat 5 & 6 until root node is reached

@group(0) @binding(0) var<storage, read_write> node_minima: array<atomic<u32>>;
@group(0) @binding(0) var<storage, read_write> node_maxima: array<atomic<u32>>;


@compute
@workgroup_size(64, 1, 1)
fn map_bricks_to_leaf_nodes(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // todo: check for out of bounds

    // todo: check if order is correct
    let page_directory_shape = vec3<u32>(
        page_directory_meta.max_resolutions,
        page_directory_meta.max_channels,
        1
    );


    let brick_id = 0;
    // todo: find nodes that overlap with this brick
    let brick_address = unpach4x8uint(brick_id);
    let channel_and_resolution = index_to_subscript(brick_address.w, page_directory_shape);

    // todo: I need the spatial offset & extent of the brick to determine which nodes are affected by this brick
    // todo: I think I need to check for padding here
    let brick_scale = brick_size / resolution.volume_size;

    // something like this?
    let bounds_min = brick_address.xyz * brick_scale;
    let bounds_max = (brick_address.xyz + brick_size) * brick_scale;

    // add all nodes the brick contributes to
    let min_node = subdivision_idx_compute_node_index(max_subdivision_index, bounds_min);
    let max_node = subdivision_idx_compute_node_index(max_subdivision_index, bounds_max);
    for (var i = min_node; i <= max_node; i += 1) {
        let node_index = to_multichannel_index(
            subdivision_idx_local_node_index(max_subdivision_index, i),
            num_channels,
            channel_index
        );
        // todo: update partially mapped
        // todo: next_level_update_node_indices needs to be reset at some point (or work with timestamps)
        next_level_update_node_indices[node_index] = u32(true);
    }

    // todo: compute min & max only if the brick has not been uploaded before
}

@compute
@workgroup_size(64, 1, 1)
fn set_up_next_level_update(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // todo: this will be in a struct
    let subidivision_index = 0;

    // todo: where does this come from? is a constant across all passes -> can use global uniform buffer
    let num_channels = 0; // the maximum number of representable channels in the octree

    let num_nodes_in_level = subdivision_idx_num_nodes_in_subdivision(subdivision_index) * num_channels;

    if (global_invocation_id.x >= num_nodes_in_level) {
        return;
    }

    let subdivision_local_multi_channel_node_index = global_invocation_id.x;
    let update_node = next_level_update_node_indices[subdivision_local_multi_channel_node_index] != 0;
    if (update_node) {
        // todo: where is num_update_nodes?
        // todo: when is num_update_nodes set to 0?
        let index = atomicAdd(num_update_nodes, 1);

        let channel_index = subdivision_local_multi_channel_node_index % num_channels;
        let channel_local_node_index = (subdivision_local_multi_channel_node_index - channel_index) / num_channels;
        update_node_indices[index] = to_multichannel_node_index(
            subdivision_idx_global_node_index(subdivision_index, channel_local_node_index),
            num_channels,
            channel_index
        );
        atomicMax(next_level_update_indirect.workgroup_count_x, max(index / 64, 1));
    }
}

@compute
@workgroup_size(64, 1, 1)
fn update_octree_node(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    // todo: num_update_nodes may be in a struct
    if (global_invocation_id.x >= num_update_nodes) {
        return;
    }

    // todo: this will be in a struct
    let subidivision_index = 0;
    let num_channels = 0; // the maximum number of representable channels in the octree
    let channel_index = 0; // the index of this thread's channel

    // todo: will node indices be global, level local or channel local
    let node_index = update_node_indices[global_invocation_id.x];

    let global_node_index = node_index;
    let channel_local_node_index = node_index;

    // todo: to update, we need to check all children of the node to update
    // todo: check if mapped or partially mapped change in the update
    // todo: check if min and max change in the update
    let node = node_idx_load_global(global_node_index);

    var minimum = 255;
    var maximum = 0;
    var mapped_resolutions = 255;
    var partially_mapped_resolutions = 0;

    let num_child_nodes = subdivision_idx_get_children_per_node(subdivision_index);
    var child_index = to_multichannel_node_index(
        subdivision_idx_first_child_index(subdivision_index, channel_local_node_index),
        num_channels,
        channel_index
    );
    for (var i = 0; i < num_child_nodes; i += 1) {
        let child_node = node_idx_load_global(child_index);

        minimum = min(minimum, node_get_min(child_node));
        maximum = max(maximum, node_get_max(child_node));
        mapped_resolutions &= node_get_mapped_resolutions(child_node);
        partially_mapped_resolutions |= node_get_partially_mapped_resolutions(child_node);

        child_index += num_channels;
    }

    let new_node = node_new(minimum, maximum, mapped_resolutions, partially_mapped_resolutions);
    let changed = node == new_node;
    if (changed) {
        node_idx_store_global(global_node_index, new_node);
        if (subdivision_index > 0) {
            let parent_node_index = to_multichannel_node_index(
                subdivision_idx_local_parent_node_index(subdivision_index, channel_local_node_index),
                num_channels,
                channel_index
            );
            // so the things stored here are local to subdivision but multichannel
            next_level_update_node_indices[parent_node_index] = u32(true);
        }
    }
}

@compute
@workgroup_size(64, 1, 1)
fn process_brick_values(@builtin(global_invocation_id) global_id: uint3) {
    // todo: for n values in brick, compute min and max and update octree leaf nodes

    // todo: compute first node index
    var current_node_index = 0;

    // todo: compute shape of the 4x4x4 volume that falls into the first node
    var current_node = vec3<u32>();

    for (var processed = 0; processed < 64;) {
        // todo: compute offset of the next node if multiple nodes are in the 4x4x4 region handled by this thread
        let next_node = vec3<u32>(4,4,4);

        var current_min = 255;
        var current_max = 0;
        for (; current_node.z < next_node.z; current_node.z += 1) {
        for (; current_node.y < next_node.y; current_node.x += 1) {
        for (; current_node.x < next_node.x; current_node.y += 1) {
            // todo: get value from brick cache
            let value = u32(0.0 * 255.0);
            current_min = min(current_min, value);
            current_max = max(current_max, value);
        }}}

        // todo: implement (atomicMin & atomicMax, or actually CAS because these are just parts of a u32...or use an extra buffer of vec4<u32> that needs to be processed afterwards)
        maybe_set_node_min_max(current_node_index, current_min, current_max);
        processed += current_node.x * current_node.y * current_node.z;
        current_node = next_node;
    }
}


