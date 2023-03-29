// This interface requires the following identifiers to be defined:
// bindings:
//  - channel_settings: array<ChannelSettings>;
//  - page_directory_meta: PageDirectoryMeta
//  - page_table_meta: PageTableMetas;
//  - page_directory: texture_3d<u32>;
//  - brick_usage_buffer: texture_storage_3d<r32uint, write>;
//  - request_buffer: texture_storage_3d<r32uint, write>;

@include(channel_settings)
@include(page_table)
@include(page_table_util)
@include(type_alias)
@include(util)
@include(volume_accessor)

fn _volume_acessor__compute_lod(distance: f32, channel: u32, lod_factor: f32) -> u32 {
    let highest_res = channel_settings[channel].max_lod;
    let lowest_res = channel_settings[channel].min_lod;
    return select_level_of_detail(distance, highest_res, lowest_res, lod_factor);
}

fn _volume_accessor__get_node(ray_sample: float3, lod: u32, channel: u32, request_bricks: bool) -> Node {
    let page_table_index = compute_page_table_index(
        channel_settings[channel].page_table_index,
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
