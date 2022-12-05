@include(type_alias)
@include(util)

// this import contains bind group 0
@include(lru_update_base)

@group(1) @binding(0) var<storage, read_write> lru_updated: ListU32;

fn was_used(global_id: u32) -> bool {
    return used_buffer.list[global_id] == 1u;
}

fn compute_new_index(global_id: u32) -> u32 {
    let offset = offsets.list[global_id];
    if (was_used(global_id)) {
        return offset - 1u;
    } else {
        let num_used_total = atomicLoad(&num_used_entries.num);
        return global_id + num_used_total - offset;
    }
}

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: uint3) {
    let global_id = global_invocation_id.x;
    if (is_out_of_bounds(global_id)) {
        return;
    }

    let lru_entry = lru_cache.list[global_id];
    let index = compute_new_index(global_id);

    lru_updated.list[index] = lru_entry;
}
