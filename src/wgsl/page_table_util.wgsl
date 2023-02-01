fn pt_get_volume_size(page_table_index: u32) -> vec3<u32> {
    return page_table_meta.metas[page_table_index].volume_size;
}

fn pt_to_local_page_address(page_table_index: u32, local_page_address: uint3) -> vec3<u32> {
    let page_table_meta = page_table_meta.metas[page_table_index];
    return page_table_meta.page_table_offset + local_page_address;
}

/// Computes the address of a page w.r.t. the page table table's offset for a given position in the unit cube ([0,1]^3)
/// The result is in range [0, page_table.extent[i] - 1], i = 0,1,2.
fn pt_compute_local_page_address(page_table_index: u32, position: float3) -> uint3 {
    let page_table_meta = page_table_meta.metas[page_table_index];
    return min(
        uint3(floor(float3(page_table_meta.page_table_extent) * position)),
        page_table_meta.page_table_extent - uint3(1u)
    );
}

/// Computes the address of a page in a page directory for a given position in the unit cube ([0,1]^3).
/// The result is in range [page_table.offset[i], page_table.offset[i] + page_table.extent[i] - 1], i = 0,1,2.
fn pt_compute_page_address(page_table_index: u32, position: float3) -> uint3 {
    let page_table_meta = page_table_meta.metas[page_table_index];
    return page_table_meta.page_table_offset + pt_compute_local_page_address(page_table_index, position);
}

fn pt_compute_cache_address(page_table_index: u32, position: float3, page: PageTableEntry) -> uint3 {
    let page_table_meta = page_table_meta.metas[page_table_index];
    let brick_size = page_table_meta.brick_size;
    let volume_size = page_table_meta.volume_size;
    return page.location + uint3(floor(position * float3(volume_size))) % brick_size;
}

// todo: this could be a per-resolution constant
/// Computes the ratio of filled voxels to total voxels in the page table.
// The result is in range [0, 1]^3.
fn pt_compute_volume_to_padded(page_table_index: u32) -> float3 {
    let page_table_meta = page_table_meta.metas[page_table_index];
    let volume_size = float3(page_table_meta.volume_size);

    let extent = page_table_meta.page_table_extent;
    let brick_size = page_table_meta.brick_size;
    let padded_size = float3(brick_size * extent);

    return volume_size / padded_size;
}

// this constructs a multi res page table index where the level and each component in the page table are
// represented as 8 bit unsigned integers - this enforces a limit on page table dimensions, namely that
// they must not contain more than [0,255]^3 bricks
fn pt_compute_brick_id(page_table_index: u32, global_id: uint3, level: u32) -> u32 {
    let page_table_meta = page_table_meta.metas[page_table_index];
    let local_brick_address = global_id - page_table_meta.page_table_offset;
    return pack4x8uint(uint4(local_brick_address, level));
}
