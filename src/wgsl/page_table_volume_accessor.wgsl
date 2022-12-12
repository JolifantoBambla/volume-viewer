// This interface requires the following identifiers to be defined:
// bindings:
//  - channel_settings_list: ChannelSettingsList;
//  - page_directory_meta: PageDirectoryMeta
//  - page_table_meta: PageTableMetas;
//  - page_directory: texture_3d<u32>;
//  - brick_usage_buffer: texture_storage_3d<r32uint, write>;
//  - request_buffer: texture_storage_3d<r32uint, write>;

@include(volume_accessor)
@include(util)

fn _volume_acessor__compute_lod(distance: f32, channel: u32, lod_factor: f32) -> u32 {
    let highest_res = channel_settings_list[channel].max_lod;
    let lowest_res = channel_settings_list[channel].min_lod;
    return select_level_of_detail(distance, highest_res, lowest_res, lod_factor);
}

fn _volume_accessor__get_node(lod: u32, channel: u32, request_bricks: bool) -> Node {
    let page_table_index = compute_page_table_index(
        channel_settings_list.channels[channel].page_table_index,
        lod
    );
    // todo: remove this cloning stuff
    let page_table = clone_page_table_meta(page_table_index);

    // todo: let compute_volume_to_padded take a page_table index
    // todo: think about this more carefully - does that change the ray or not?
    let position_corrected = p * compute_volume_to_padded(page_table);
    let page_address = compute_page_address(page_table, position_corrected);
    let page = get_page(page_address);

    if (page.flag == UNMAPPED) {
        var requested_brick = false;
        if (request_bricks) {
            // todo: maybe request lower res as well?
            request_brick(int3(page_address));
            requested_brick = true;
        }
        return Node(
            false,
            0.0,
            false,
            vec3(),
            requested_brick
        );
    } else if (page.flag == MAPPED) {
        report_usage(int3(page.location / brick_size));

        let sample_location = normalize_cache_address(
            compute_cache_address(page_table, p, page)
        );
        return Node(
            false,
            0.0,
            true,
            sample_location,
            false
        );
    }
}
