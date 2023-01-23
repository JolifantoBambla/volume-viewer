@include(util)

// CHANNEL AGNOSTIC INDEX OPERATIONS
// ----------------------------------------------------------------------------
// All indices computed by the following functions assume there is a single
// continuous list of nodes belonging to a single channel sorted by subdivision
// level (from lowest to highest subdivision)

fn num_nodes_in_subdivision(subdivision_index: u32) -> u32 {
    let shape = volume_subdivisions[subdivision_index].shape;
    return shape.x * shape.y * shape.z;
}

fn next_subdivision_offset(subdivision_index: u32) -> u32 {
    return volume_subdivisions[subdivision_index].node_offset + num_nodes_in_subdivision(subdivision_index);
}

fn local_node_index(subdivision_index: u32, node_index: u32) -> u32 {
    return node_index - volume_subdivisions[subdivision_index].node_offset;
}

fn compute_node_index(subdivision_index: u32, normalized_address: vec3<f32>) -> u32 {
    let shape = volume_subdivisions[subdivision_index].shape;
    return volume_subdivisions[subdivision_index].node_offset +
        subscript_to_index(
            normalized_address_to_subscript(normalized_address, shape),
            shape
        );
}

fn first_child_index(subdivision_index: u32, node_index: u32) -> u32 {
    return next_subdivision_offset(subdivision_index) +
        local_node_index(subdivision_index, node_index) *
        volume_subdivisions[subdivision_index].children_per_node;
}

