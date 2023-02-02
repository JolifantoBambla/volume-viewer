// REQUIRES THE FOLLOWING BINDINGS:
// - var<uniform> page_directory_meta: PageDirectoryMeta;

@include(page_table)
@include(util)

fn page_directory_shape() -> vec3<u32> {
    return vec3<u32>(page_directory_meta.max_channels, page_directory_meta.max_resolutions, 1);
}

fn page_directory_compute_page_table_index(channel: u32, lod: u32) -> u32 {
    return subscript_to_index(vec3<u32>(channel, lod, 0), page_directory_shape());
}

fn page_directory_compute_page_table_subscript(index: u32) -> vec2<u32> {
    return index_to_subscript(index, page_directory_shape()).xy;
}
