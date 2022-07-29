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

// todo: fix all these alignment issues here and on cpu side!
// todo: I probably won't need all this stuff -> find out what is needed and kick out the rest

struct VolumeResolutionMeta {
    @size(16) volume_size: uint3,
    @size(16) padded_volume_size: uint3,
    @size(16) scale: float3,
}

struct PageTableMeta {
    @size(16) offset: uint3,
    @size(16) extent: uint3,
    volume_meta: VolumeResolutionMeta,
}

/*
struct PageDirectoryMeta {
    @size(16) brick_size: uint3,
    @size(16) extent: uint3,
    resolutions: array<PageTableMeta>,
}
*/

struct ResolutionMeta {
    @size(16) brick_size: uint32,
    @size(16) page_table_offset: uint32,
    @size(16) page_table_extent: uint32,
    @size(16) volume_size: uint32,
}

struct PageDirectoryMeta {
    resolutions: array<ResolutionMeta>,
}

fn pt_canonical_to_voxel(page_table_meta: PageDirectoryMeta, p: float3, level: u32) -> int32 {
    return int3(p * page_directory_meta.resolutions[level].volume_meta.volume_size);
}

fn pt_compute_lod(page_table_meta: PageDirectoryMeta, distance_to_camera: f32) -> u32 {
    // todo: select based on distance to camera
    return min(arrayLength(page_table_meta.resolutions), 0u);
}
