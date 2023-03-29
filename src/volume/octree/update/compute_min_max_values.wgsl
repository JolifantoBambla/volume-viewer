@include(cache_update_meta)
@include(dispatch_indirect)
@include(multichannel_octree_util)
@include(page_directory_util)
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
@group(0) @binding(3) var page_directory: texture_3d<u32>;
@group(0) @binding(4) var brick_cache: texture_3d<f32>;

// (read-only) cache update bind group
@group(1) @binding(0) var<storage> cache_update_meta: CacheUpdateMeta;
@group(1) @binding(1) var<storage> new_brick_ids: array<u32>;

// (read-write) output minima & maxima
@group(2) @binding(0) var<storage, read_write> node_helper_buffer_a: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> node_helper_buffer_b: array<atomic<u32>>;

const THREAD_BLOCK_SIZE: vec3<u32> = vec3<u32>(2, 2, 2);

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let brick_size = page_directory_meta.brick_size;
    let processing_size = brick_size / THREAD_BLOCK_SIZE;
    let threads_per_brick = processing_size.x * processing_size.y * processing_size.z;

    let global_id = global_invocation_id.x;
    let brick_index = global_id / threads_per_brick;

    if (brick_index >= cache_update_meta.num_mapped_first_time) {
        return;
    }

    let local_id = global_id - brick_index * threads_per_brick;
    let thread_block_offset = index_to_subscript(local_id, processing_size) * THREAD_BLOCK_SIZE;

    let brick_id = new_brick_ids[brick_index];
    let unpacked_brick_id = unpack4x8uint(brick_id);
    let local_page_address = unpacked_brick_id.xyz;
    let page_table_index = unpacked_brick_id.w;
    let channel_index = page_directory_compute_page_table_subscript(page_table_index).x;
    let page_address = pt_to_global_page_address(page_table_index, local_page_address);
    let page = page_directory_get_page(page_address.xyz);
    let brick_cache_offset = page.location;

    let offset = brick_cache_offset + thread_block_offset;

    let volume_offset = local_page_address * brick_size + thread_block_offset;
    let volume_size = pt_get_volume_size(page_table_index);
    if (any(volume_offset >= volume_size)) {
        return;
    }

    //let page_table_extent = pt_get_page_table_extent(page_table_index);
    //let volume_to_padded = pt_compute_volume_to_padded(page_table_index);

    let subdivision_index = subdivision_get_leaf_node_level_index();
    let num_channels = page_directory_meta.max_channels;

    var current_multichannel_local_index = to_multichannel_node_index(
        subdivision_idx_compute_local_node_index(
            subdivision_index,
            subscript_to_normalized_address(volume_offset, volume_size)
        ),
        num_channels,
        channel_index
    );

    var current_min: u32 = 255;
    var current_max: u32 = 0;
    for (var z: u32 = 0; z < THREAD_BLOCK_SIZE.z; z += 1) {
        for (var x: u32 = 0; x < THREAD_BLOCK_SIZE.x; x += 1) {
            for (var y: u32 = 0; y < THREAD_BLOCK_SIZE.y; y += 1) {
                let position = vec3<u32>(x, y, z);

                let volume_position = volume_offset + position;
                let out_of_bounds = any(volume_position >= volume_size);

                if (!out_of_bounds) {
                    let node_index = to_multichannel_node_index(
                        subdivision_idx_compute_local_node_index(
                            subdivision_index,
                            subscript_to_normalized_address(volume_offset, volume_size)
                        ),
                        num_channels,
                        channel_index
                    );

                    // if we've entered a new node, we commit our intermediate results before we move on
                    if (node_index != current_multichannel_local_index) {
                        if (current_min < 255) {
                            atomicMin(&node_helper_buffer_a[current_multichannel_local_index], current_min);
                        }
                        if (current_max > 0) {
                            atomicMax(&node_helper_buffer_b[current_multichannel_local_index], current_max);
                        }
                        current_multichannel_local_index = node_index;
                        current_min = 255;
                        current_max = 0;
                    }

                    let chache_position = offset + position;
                    var value = min(u32(textureLoad(brick_cache, int3(chache_position), 0).x * 255.0), 255);
                    if (page.flag == EMPTY) {
                        value = 0;
                    }
                    current_min = min(current_min, value);
                    current_max = max(current_max, value);
                }
            }
        }
    }
    if (current_min < 255) {
        atomicMin(&node_helper_buffer_a[current_multichannel_local_index], current_min);
    }
    if (current_max > 0) {
        atomicMax(&node_helper_buffer_b[current_multichannel_local_index], current_max);
    }
}
