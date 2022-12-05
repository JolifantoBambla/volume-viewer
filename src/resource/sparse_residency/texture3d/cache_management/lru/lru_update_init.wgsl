@include(timestamp)
@include(type_alias)
@include(util)

// this import contains bind group 0
@include(lru_update_base)

@group(1) @binding(0) var usage_buffer: texture_3d<u32>;
@group(1) @binding(1) var<uniform> timestamp: Timestamp;

fn was_used(lru_entry: u32) -> bool {
    let buffer_size = uint3(textureDimensions(usage_buffer));
    let usage_index = int3(index_to_subscript(lru_entry, buffer_size));
    return timestamp.now == u32(textureLoad(usage_buffer, usage_index, 0).r);
}

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: uint3) {
    let global_id = global_invocation_id.x;
    if (is_out_of_bounds(global_id)) {
        return;
    }

    let lru_entry = lru_cache.list[global_id];
    let used = was_used(lru_entry);

    offsets.list[global_id] = u32(used);
    used_buffer.list[global_id] = u32(used);

    if (used) {
        atomicAdd(&num_used_entries.num, 1u);
    }
}
