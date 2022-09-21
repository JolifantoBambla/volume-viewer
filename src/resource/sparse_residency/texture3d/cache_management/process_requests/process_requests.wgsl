
@include(gpu_list)
@include(page_table)
@include(type_alias)
@include(timestamp)
@include(util)

@group(0) @binding(0) var<storage> page_table_meta: PageDirectoryMeta;
@group(0) @binding(1) var request_buffer: texture_3d<u32>;
@group(0) @binding(2) var<uniform> timestamp: Timestamp;

@group(1) @binding(0) var<storage, read_write> ids: ListU32;
@group(1) @binding(1) var<storage, read_write> ids_meta: ListMeta;

@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    let buffer_size = uint3(textureDimensions(request_buffer));
    if any(buffer_size < global_id) {
        return;
    }

    if atomicLoad(&ids_meta.fill_pointer) >= ids_meta.capacity {
        return;
    }

    if timestamp.now == u32(textureLoad(request_buffer, int3(global_id), 0).r) {
        for (var i = 0u; i < arrayLength(&page_table_meta.resolutions); i += 1u) {
            let res = page_table_meta.resolutions[i];
            if all(global_id >= res.page_table_offset) && all(global_id < (res.page_table_offset + res.page_table_extent)) {
                let index = atomicAdd(&ids_meta.fill_pointer, 1u);
                if index > ids_meta.capacity {
                    return;
                }

                // this constructs a multi res page table index where the level and each component in the page table are
                // represented as 8 bit unsigned integers - this enforces a limit on page table dimensions, namely that
                // they must not contain more than [0,253]^3 bricks
                let brick_address = global_id - res.page_table_offset;

                let brick_id = pack4x8uint(uint4(brick_address, i));

                ids.list[index] = brick_id;
                return;
            }
        }
    }
}