// REQUIRES THE FOLLOWING BINDINGS:
// - var<storage> volume_subdivisions: array<VolumeSubdivision>

@include(util)

// CHANNEL AGNOSTIC INDEX OPERATIONS
// ----------------------------------------------------------------------------
// All indices computed by the following functions assume there is a single
// continuous list of nodes belonging to a single channel sorted by subdivision
// level (from lowest to highest subdivision)

fn subdivision_idx_get_shape(subdivision_index: u32) -> u32 {
    return volume_subdivisions[subdivision_index].shape;
}

fn subdivision_idx_get_node_offset(subdivision_index: u32) -> u32 {
    return volume_subdivisions[subdivision_index].node_offset;
}

fn subdivision_idx_get_children_per_node(subdivision_index: u32) -> u32 {
    return volume_subdivisions[subdivision_index].children_per_node;
}

fn subdivision_idx_num_nodes_in_subdivision(subdivision_index: u32) -> u32 {
    let shape = subdivision_idx_get_shape(subdivision_index);
    return shape.x * shape.y * shape.z;
}

fn subdivision_idx_next_subdivision_offset(subdivision_index: u32) -> u32 {
    return subdivision_idx_get_node_offset(subdivision_index) +
        subdivision_idx_num_nodes_in_subdivision(subdivision_index);
}

fn subdivision_idx_local_node_index(subdivision_index: u32, global_node_index: u32) -> u32 {
    return global_node_index - subdivision_idx_get_node_offset(subdivision_index);
}

fn subdivision_idx_global_node_index(subdivision_index: u32, local_node_index: u32) -> u32 {
    return local_node_index + subdivision_idx_get_node_offset(subdivision_index);
}

fn subdivision_idx_compute_local_node_index(subdivision_index: u32, normalized_address: vec3<f32>) -> u32 {
    let shape = subdivision_idx_get_shape(subdivision_index);
    return subscript_to_index(
        normalized_address_to_subscript(normalized_address, shape),
        shape
    );
}

fn subdivision_idx_compute_node_index(subdivision_index: u32, normalized_address: vec3<f32>) -> u32 {
    return subdivision_idx_get_node_offset(subdivision_index) +
        subdivision_idx_compute_local_node_index(subdivision_index, normalized_address);
}

fn subdivision_idx_first_child_index(subdivision_index: u32, node_index: u32) -> u32 {
    return subdivision_idx_next_subdivision_offset(subdivision_index) +
        subdivision_idx_local_node_index(subdivision_index, node_index) *
        subdivision_idx_get_children_per_node(subdivision_index);
}

fn subdivision_idx_local_parent_node_index(subdivision_index: u32, node_index: u32) -> u32 {
    let parent_subdivision_index = subdivision_index - 1;
    return subdivision_idx_local_node_index(subdivision_index, node_index) / subdivision_idx_get_children_per_node(parent_subdivision_index);
}

fn subdivision_idx_parent_node_index(subdivision_index: u32, node_index: u32) -> u32 {
    let parent_subdivision_index = subdivision_index - 1;
    let local_parent_node_index = subdivision_idx_local_parent_node_index(subdivision_index, node_index);
    return subdivision_idx_get_node_offset(parent_subdivision_index) + local_parent_node_index;
}
