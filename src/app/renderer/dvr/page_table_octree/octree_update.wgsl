@include(volume_subdivision)
@include(volume_subdivision_util)

@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;

// MULTICHANNEL INDEX OPERATIONS
// ----------------------------------------------------------------------------

// Computes the global node index in a list of interleaved octree nodes from a local one, i.e., an index that is unaware
// of other channels in the same octree
fn to_multichannel_node_index(local_node_index: u32, num_channels: u32, channel_index: u32) -> u32 {
    return local_node_index * num_channels + channel_index;
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


