// MULTICHANNEL INDEX OPERATIONS
// ----------------------------------------------------------------------------

// Computes the global node index in a list of interleaved octree nodes from a local one, i.e., an index that is unaware
// of other channels in the same octree
fn to_multichannel_node_index(local_node_index: u32, num_channels: u32, channel_index: u32) -> u32 {
    return local_node_index * num_channels + channel_index;
}

// todo
fn to_channel_local_node_index(multichannel_node_index: u32, num_channels: u32) -> u32 {
}
