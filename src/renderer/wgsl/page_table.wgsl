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
fn compute_local_page_address(page_table: ptr<function, PageTableMeta, read_write>, position: float3) -> uint3 {
    let pt = *page_table;
    return min(
        uint3(floor(float3(pt.page_table_extent) * position)),
        pt.page_table_extent - uint3(1u)
    );
}

/// Computes the address of a page in a page directory for a given position in the unit cube ([0,1]^3).
/// The result is in range [page_table.offset[i], page_table.offset[i] + page_table.extent[i] - 1], i = 0,1,2.
fn compute_page_address(page_table: ptr<function, PageTableMeta, read_write>, position: float3) -> uint3 {
    let pt = *page_table;
    return pt.page_table_offset + compute_local_page_address(page_table, position);
}

/// Computes the ratio of filled voxels to total voxels in the page table.
// The result is in range [0, 1]^3.
fn compute_volume_to_padded(page_table: ptr<function, PageTableMeta, read_write>) -> float3 {
    let pt = *page_table;

    let volume_size = float3(pt.volume_size);

    let extent = pt.page_table_extent;
    let brick_size = pt.brick_size;
    let padded_size = float3(brick_size * extent);

    return volume_size / padded_size;
}

struct PageDirectoryMeta {
    resolutions: array<PageTableMeta>,
}