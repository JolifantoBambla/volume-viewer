@include(cache_update_meta)
@include(multichannel_octree_util)
@include(octree_node)
@include(octree_node_util)
@include(page_directory_meta_util)
@include(page_table)
@include(page_table_util)
@include(volume_subdivision)
@include(volume_subdivision_util)
@include(util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<uniform> page_directory_meta: PageDirectoryMeta;
@group(0) @binding(2) var<storage> page_table_meta: PageTableMetas;
@group(0) @binding(3) var<storage> octree_nodes: array<u32>;

// (read-only) cache update bind group
@group(1) @binding(0) var<storage> cache_update_meta: CacheUpdateMeta;
@group(1) @binding(1) var<storage> mapped_brick_ids: array<u32>;

// (read-write) output resolution mapping
@group(2) @binding(0) var<storage, read_write> node_helper_buffer_a: array<atomic<u32>>;

// Maps bricks to octree leaf nodes and marks their local index in node_helper_buffer_a as updated with the updated
// non-zero bitmask of partially mapped resolutions
@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let brick_id_index = global_invocation_id.x;
    if (brick_id_index >= cache_update_meta.num_mapped) {
        return;
    }

    let subdivision_index = subdivision_get_leaf_node_level_index();
    let num_channels = page_directory_meta.max_channels;
    let brick_size = page_directory_meta.brick_size;

    let brick_id = mapped_brick_ids[brick_id_index];
    let unpacked_brick_id = unpack4x8uint(brick_id);
    let local_page_address = unpacked_brick_id.xyz;
    let page_table_index = unpacked_brick_id.w;
    let volume_size = pt_get_volume_size(page_table_index);

    let volume_min = local_page_address * brick_size;
    let node_min = subdivision_idx_compute_subscript(
        subdivision_index,
        subscript_to_normalized_address(local_page_address, pt_get_page_table_extent(page_table_index))
    );
    let node_max = subdivision_idx_compute_subscript(
        subdivision_index,
        subscript_to_normalized_address(min(local_page_address * brick_size + brick_size, volume_size), volume_size)
    );

    let channel_and_resolution = page_directory_compute_page_table_subscript(page_table_index);
    let channel_index = channel_and_resolution.x;
    let resolution_mask = node_make_mask_for_resolution(channel_and_resolution.y);

    for (var x = node_min.x; x <= node_max.x; x += 1) {
        for (var y = node_min.y; y <= node_max.y; y += 1) {
            for (var z = node_min.z; z <= node_max.z; z += 1) {
                let single_channel_local_index = subdivision_idx_subscript_to_local_index(subdivision_index, vec3<u32>(x, y, z));
                let multichannel_local_index = to_multichannel_node_index(
                    single_channel_local_index,
                    num_channels,
                    channel_index
                );
                let multichannel_global_index = to_multichannel_node_index(
                    subdivision_idx_global_node_index(subdivision_index, single_channel_local_index),
                    num_channels,
                    channel_index
                );
                let node = octree_nodes[multichannel_global_index];
                let partially_mapped_bitmask = node_get_partially_mapped_resolutions(node);
                if ((partially_mapped_bitmask & resolution_mask) == 0u) {
                    //atomicAdd(&node_helper_buffer_b[multichannel_local_index], 1);
                    //atomicOr(&node_helper_buffer_b[multichannel_local_index], resolution_mask);
                    atomicOr(&node_helper_buffer_a[multichannel_local_index], insertBits(0, resolution_mask, PARTIALLY_MAPPED_RESOLUTIONS_OFFSET, PARTIALLY_MAPPED_RESOLUTIONS_COUNT));
                }
            }
        }
    }
}
