@include(dispatch_indirect)
@include(multichannel_octree_util)
@include(octree_node)
@include(octree_node_util)
@include(page_directory_util)
@include(page_table)
@include(page_table_util)
@include(volume_subdivision)
@include(volume_subdivision_util)
@include(util)

// with histogram/bitmask:
// 1. compute min max:
//  - compute bitmask for values in block
//  - read node and check if bitmask is different
//  - if so, atomicOr in helper buffer
// 2. process mapped: (maybe exclude first mapped from this process and do a combined one in min max pass for those)
//  - compute bitmask for resolution
//  - read node and check if bitmask is different
//  - if so, atomicOr in helper buffer (same as min & max)
// 3. process helper buffer:
//  - if non-zero: Or in node buffer
//  - set helper buffer 0
//  - mark parent node in helper buffer b
// 4. process parent node buffer:
//  - if non-zero add node in helper buffer b to a, etc.
//  - set helper buffer b 0
// 5. process parent nodes:
//  - check child nodes and do Or
//  - if changed, set parent node in helper buffer b to true
// repeat 4. and 5. until root node

// with min max:
// 1. compute min max:
//  - compute min and store in helper a (atomicMin)
//  - compute max and store in helper b (atomicMax)
// 2. update min max:
//  - if min or max in helper buffers is not default value
//  - update node
//  - mark in helper buffer b if min max changed
//  - set helper buffer a to 255
// 2. process mapped:
//  - compute bitmask for resolution
//  - read node and check if bitmask is different
//  - if so, atomicOr in helper buffer b
// 3. process helper buffers:
//  - if helper buffer b non-zero: update node & collect parent node in helper buffer a
//  - set helper buffer b to 0
// 4. process parent nodes:
//  - check child nodes and update node
//  - if changed, set parent node in helper buffer b to true
//  - set helper buffer a to 255
// 5. process parent node buffer:
//  - if helper buffer b non-zero: atomicAdd index & add node index to helper buffer a
//  - set helper buffer b 0
// repeat 4. and 5. until root node



// prerequisites:
// todo: compute_min_max_values: set minima to 255
// todo: compute_min_max_values: set maxima to 0
// todo: process_mapped_bricks: set node_helper_buffer_a to 0
// todo: set_up_next_level_update: set num_nodes_next_level to 0
// todo: set_up_next_level_update: set next_level_update_indirect.workgroup_count_x to 0

@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<storage, read_write> octree_nodes: array<u32>;

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
