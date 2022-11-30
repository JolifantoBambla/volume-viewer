@include(gpu_list)
@include(timestamp)
@include(type_alias)
@include(util)

struct NumUsedEntries {
    num: atomic<u32>,
}

@group(0) @binding(0) var usage_buffer: texture_3d<u32>;
@group(0) @binding(1) var<uniform> timestamp: Timestamp;
@group(0) @binding(2) var<storage, read_write> lru_cache: ListU32;
@group(0) @binding(3) var<storage, read_write> num_used_entries: NumUsedEntries;
@group(0) @binding(4) var<storage, read_write> offsets: ListU32;

@group(1) @binding(0) var<storage, read_write> lru_updated: ListU32;

const WORKGROUP_SIZE: u32 = 256;

fn get_num_entries() -> u32 {
    return arrayLength(&lru_cache.list);
}

fn is_out_of_bounds(global_id: u32) -> bool {
    return global_id >= get_num_entries();
}

fn is_used(lru_entry: u32) -> bool {
    let buffer_size = uint3(textureDimensions(usage_buffer));
    let usage_index = int3(index_to_subscript(lru_entry, buffer_size));
    return timestamp.now == u32(textureLoad(usage_buffer, usage_index, 0).r);
}

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn initialize_offsets(@builtin(global_invocation_id) global_invocation_id: uint3) {
    let global_id = global_invocation_id.x;
    if (is_out_of_bounds(global_id)) {
        return;
    }

    let lru_entry = lru_cache.list[global_id];
    let used = is_used(lru_entry);

    offsets.list[global_id] = u32(used);
    if (used) {
        atomicAdd(&num_used_entries.num, 1u);
    }
}

fn compute_new_index(global_id: u32, used: bool) -> u32 {
    let offset = offsets.list[global_id];
    if (used) {
        return offset - 1u;
    } else {
        let num_used_total = atomicLoad(&num_used_entries.num);
        return global_id + num_used_total - offset;
    }
}

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn update_lru(@builtin(global_invocation_id) global_invocation_id: uint3) {
    let global_id = global_invocation_id.x;
    if (is_out_of_bounds(global_id)) {
        return;
    }

    let lru_entry = lru_cache.list[global_id];
    let index = compute_new_index(global_id, is_used(lru_entry));

    lru_updated.list[index] = lru_entry;
}
