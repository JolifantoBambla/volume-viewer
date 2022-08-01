@include(type_alias)
@include(page_table)

struct Ids {
    ids: array<u32>,
}

struct Params {
    max_requests: u32,
    timestamp: u32,
}

struct Counters {
    num_requested_ids: atomic<u32>,
}

@group(0) @binding(1) var request_buffer: texture_3d<u32>;
@group(0) @binding(4) var<storage> ids: Ids;

@group(0) @binding(0) var<storage> page_table_meta: PageDirectoryMeta;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage> counters: Counters;

@stage(compute)
@workgroup_size(16, 16, 16)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    let buffer_size = uint3(textureDimensions(request_buffer));
    if any(buffer_size < global_id) {
        return;
    }

    if atomicLoad(&counters.num_requested_ids) >= params.max_requests {
        return;
    }

    if params.timestamp == u32(textureLoad(request_buffer, int3(global_id), 0).r) {
        let index = atomicAdd(&counters.num_requested_ids, 1u);
        if index > params.max_requests {
            return;
        }

        for (var i = 0u; i < arrayLength(page_table_meta.resolutions); i += 1u) {
            let res = page_table_meta.resolutions[i];
            if all(global_id > res.page_table_offset) && all(global_id < (res.page_table_offset + res.page_table_extent)) {
                // this constructs a multi res page table index where the level and each component in the page table are
                // represented as 8 bit unsigned integers - this enforces a limit on page table dimensions, namely that
                // they must not contain more than [0,253]^3 bricks
                let brick_address = (global_id - res.page_table_offset) / res.page_table_extent;
                let brick_id = (brick_address.x << 24) + (brick_address.y << 16) + (brick_address << 8) + i;

                ids.ids[index] = brick_id;
                break;
            }
        }
    }
}