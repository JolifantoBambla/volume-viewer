@include(octree_node)
@include(volume_subdivision_util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;

// (read-only) cache update bind group
@group(1) @binding(0) var<storage, read_write> node_helper_buffer_a: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> node_helper_buffer_b: array<atomic<u32>>;

// (read-write) output nodes
@group(2) @binding(0) var<storage, read_write> octree_nodes: array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn update_node_min_max_values(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let subdivision_index = arrayLength(volume_subdivisions) - 1;

    let local_node_index = global_invocation_id.x;
    if (local_node_index >= subdivision_idx_num_nodes_in_subdivision(subdivision_index)) {
        return;
    }

    let minimum = node_helper_buffer_a[local_node_index];
    let maximum = node_helper_buffer_b[local_node_index];

    let global_node_index = subdivision_idx_global_node_index(subdivision_index, local_node_index);
    let node = octree_nodes[global_node_index];
    node_set_min(node, min(minimum, node_get_min(node)));
    node_set_max(node, max(maximum, node_get_max(node)));
    octree_nodes[global_node_index] = node;
}
