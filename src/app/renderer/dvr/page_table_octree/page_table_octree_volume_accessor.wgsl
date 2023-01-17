// This interface requires the following identifiers to be defined:
// bindings:
//  - channel_settings_list: ChannelSettingsList;
//  - page_directory_meta: PageDirectoryMeta
//  - page_table_meta: PageTableMetas;
//  - page_directory: texture_3d<u32>;
//  - brick_usage_buffer: texture_storage_3d<r32uint, write>;
//  - request_buffer: texture_storage_3d<r32uint, write>;

@include(page_table)
@include(page_table_util)
@include(type_alias)
@include(util)
@include(volume_accessor)

struct OctreeSubdivision {
    size: vec3<u32>,
    node_offset: u32,
}

@group(2) @binding(0) var<storage> octree_subdivision: array<OctreeSubdivision>;
@group(2) @binding(1) var<storage> octree_nodes: array<u32>;

fn _volume_acessor__compute_lod(distance: f32, channel: u32, lod_factor: f32) -> u32 {
    let highest_res = 0;
    let lowest_res = arrayLength(octree_subdivision);
    let lod = select_level_of_detail(distance, highest_res, lowest_res, lod_factor);
    // note: lods are sorted in other direction here
    return highest_res - lod;
}

fn _volume_accessor__get_node(ray_sample: float3, lod: u32, channel: u32, request_bricks: bool) -> Node {
    // map channel to channel index in octree (or have octree sorted by visible channel indices?)

    // if node at lod is mapped
    //   return node if mapped
    // var last_node = node at lod 0
    // var last_mapped_node = last_node
    // var last_lod = 0
    // var next_subtree_index
    // while last_lod != lod:
    //   if subtree is not partially mapped
    //     break
    //   last_node = subtree_node
    //   if last_node is mapped:
    //     last_mapped_node = last_node
    //   update loop variables
    // return last_mapped_node

    let page_table_index = compute_page_table_index(
        channel_settings_list.channels[channel].page_table_index,
        lod
    );

    // todo: store volume_to_padded in page table (is it already?)
    let position_corrected = ray_sample * pt_compute_volume_to_padded(page_table_index);
    let page_address = pt_compute_page_address(page_table_index, position_corrected);
    let page = get_page(page_address);

    var requested_brick = false;
    if (page.flag == UNMAPPED) {
        if (request_bricks) {
            // todo: maybe request lower res as well?
            request_brick(int3(page_address));
            requested_brick = true;
        }
    } else if (page.flag == MAPPED) {
        report_usage(int3(page.location / page_directory_meta.brick_size));

        let sample_location = normalize_cache_address(
            pt_compute_cache_address(page_table_index, ray_sample, page)
        );
        return Node(
            false,
            0.0,
            true,
            sample_location,
            false
        );
    }

    return Node(
        false,
        0.0,
        false,
        float3(),
        requested_brick
    );
}
