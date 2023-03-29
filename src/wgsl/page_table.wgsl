@include(type_alias)

alias PageTableFlag = u32;
const UNMAPPED = 0u;
const MAPPED = 1u;
const EMPTY = 2u;

// todo: this should be
// bit 0: UNMAPPED / MAPPED
// bit 1: NON_EMPTY / EMPTY
// bit 3: UNKNOWN / KNOWN
// 5 bits for something else or unused
// EMPTY implies KNOWN
// UNKNOWN implies UNMAPPED

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

struct PageTableMetas {
    metas: array<PageTableMeta>,
}

struct PageDirectoryMeta {
    // size of a brick in voxels
    @size(16) brick_size: uint3,
    max_resolutions: u32,
    max_channels: u32,
    padding1: u32,
    padding2: u32,
}