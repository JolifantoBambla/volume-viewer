@include(type_alias)

type PageTableFlag = u32;
let UNMAPPED = 0u;
let MAPPED = 1u;
let EMPTY = 2u;

struct PageTableEntry {
    // The 3D texture coordinate of the brick referenced by this `PageTableEntry` in the brick
    // cache.
    // Note: this is only valid if `flag` is `MAPPED`
    location: uint3,

    // A flag signalling if the brick referenced by this `PageTableEntry` is present (`MAPPED`),
    // not present and possible non-empty (`UNMAPPED`), or possibly present but
    // does not hold any meaningful values w.r.t. the current parameters (e.g. transfer function,
    // threshold, ...) (`EMPTY`).
    flag: PageTableFlag,
}

// Constructs a PageTableEntry from a vec4<u32>.
fn to_page_table_entry(v: uint4) -> PageTableEntry {
    return PageTableEntry(v.xyz, v.w);
}

struct PageTableMeta {
    // size of a brick in voxels
    @size(16) brick_size: uint3,

    // the page table's offset from the origin
    @size(16) page_table_offset: uint3,

    // number of bricks per dimension
    @size(16) page_table_extent: uint3,

    // number of filled voxels per dimension
    @size(16) volume_size: uint3,

    // todo: store volume_to_padded ratio?
}

/// Computes the address of a page w.r.t. the page table table's offset for a given position in the unit cube ([0,1]^3)
/// The result is in range [0, page_table.extent[i] - 1], i = 0,1,2.
fn compute_local_page_address(page_table: PageTableMeta, position: float3) -> uint3 {
    return min(
        uint3(floor(float3(page_table.page_table_extent) * position)),
        page_table.page_table_extent - uint3(1u)
    );
}

/// Computes the address of a page in a page directory for a given position in the unit cube ([0,1]^3).
/// The result is in range [page_table.offset[i], page_table.offset[i] + page_table.extent[i] - 1], i = 0,1,2.
fn compute_page_address(page_table: PageTableMeta, position: float3) -> uint3 {
    return page_table.page_table_offset + compute_local_page_address(page_table, position);
}

fn compute_cache_address(page_table: PageTableMeta, position: float3, page: PageTableEntry) -> uint3 {
    let brick_size = page_table.brick_size;
    let volume_size = page_table.volume_size;
    // todo: this could lead to -1! e.g.: max(ceil(0.8 * 40.), 1) % 32 - 1
    return page.location + (max(uint3(ceil(position * float3(volume_size))), uint3(1u)) % brick_size) - 1u;
}

// todo: this could be a per-resolution constant
/// Computes the ratio of filled voxels to total voxels in the page table.
// The result is in range [0, 1]^3.
fn compute_volume_to_padded(page_table: PageTableMeta) -> float3 {
    let volume_size = float3(page_table.volume_size);

    let extent = page_table.page_table_extent;
    let brick_size = page_table.brick_size;
    let padded_size = float3(brick_size * extent);

    return volume_size / padded_size;
}

// this constructs a multi res page table index where the level and each component in the page table are
// represented as 8 bit unsigned integers - this enforces a limit on page table dimensions, namely that
// they must not contain more than [0,255]^3 bricks
fn compute_brick_id(page_table: PageTableMeta, global_id: uint3, level: u32) -> u32 {
    let local_brick_address = global_id - page_table.page_table_offset;
    return pack4x8uint(uint4(local_brick_address, level));
}

struct PageDirectoryMeta {
    resolutions: array<PageTableMeta>,
}