// REQUIRES THE FOLLOWING BINDINGS:
// - var page_directory: texture_3d<u32>;

@include(page_table)

// todo: add mutliple virtualization levels
fn page_directory_get_page(page_address: vec3<u32>) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, vec3<i32>(page_address), 0));
}
