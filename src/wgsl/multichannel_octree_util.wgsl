// MULTICHANNEL INDEX OPERATIONS
// ----------------------------------------------------------------------------

// Computes the global node index in a list of interleaved octree nodes from a local one, i.e., an index that is unaware
// of other channels in the same octree
fn to_multichannel_node_index(single_channel_level_node_index: u32, num_channels: u32, channel_index: u32) -> u32 {
    return single_channel_level_node_index * num_channels + channel_index;
}

fn multi_local_to_single_channel_local_index(multichannel_local_index: u32, num_channels: u32) -> u32 {
    return multichannel_local_index / num_channels;
}

fn multi_local_to_channel_index(multichannel_local_index: u32, num_channels: u32) -> u32 {
    let single_channel_local_index = multi_local_to_single_channel_local_index(multichannel_local_index, num_channels);
    return multichannel_local_index - single_channel_local_index * num_channels;
}

// todo
fn to_channel_local_node_index(multichannel_node_index: u32, num_channels: u32) -> u32 {
}
