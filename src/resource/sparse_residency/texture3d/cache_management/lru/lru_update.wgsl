@include(gpu_list)
@include(timestamp)
@include(type_alias)
@include(util)

struct NumUsedEntries {
    num: u32,
}

@group(0) @binding(0) var usage_buffer: texture_3d<u32>;
@group(0) @binding(1) var<uniform> timestamp: Timestamp;
@group(0) @binding(2) var<storage, read_write> lru_cache: ListU32;
@group(0) @binding(3) var<storage, read_write> num_used_entries: NumUsedEntries;

// these resources are never used outside of this pass
@group(1) @binding(0) var<storage, read_write> scan_even: ListU32;
@group(1) @binding(1) var<storage, read_write> scan_odd: ListU32;

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    let buffer_size = uint3(textureDimensions(usage_buffer));
    let num_entries = buffer_size.x * buffer_size.y * buffer_size.z;
    let cache_entry_index = global_id.x;

    // note: we can't return here because then storageBarrier calls would be outside of uniform control flow, so what we
    // do instead is that we wrap all operations in if-else blocks checking for `OUT_OF_BOUNDS`
    let OUT_OF_BOUNDS = cache_entry_index >= num_entries;

    // initialize number of used entries
    var lru_entry = 0u;
    var used = false;
    if (!OUT_OF_BOUNDS) {
        lru_entry = lru_cache.list[cache_entry_index];
        used = timestamp.now == u32(textureLoad(usage_buffer, int3(index_to_subscript(lru_entry, buffer_size)), 0).r);
        scan_odd.list[cache_entry_index] = u32(used);
    }
    
    storageBarrier();

    // determine number of used entries to the left of the current entry including the current entry itself
    var even_pass = true;
    for (var lookup = 1u; lookup <= num_entries; lookup *= 2u) {
        if (!OUT_OF_BOUNDS) {
            if (even_pass) {
                if (cache_entry_index >= lookup) {
                    scan_even.list[cache_entry_index] = scan_odd.list[cache_entry_index] + scan_odd.list[cache_entry_index - lookup];
                } else {
                    scan_even.list[cache_entry_index] = scan_odd.list[cache_entry_index];
                }
            } else {
                if (cache_entry_index >= lookup) {
                    scan_odd.list[cache_entry_index] = scan_even.list[cache_entry_index] + scan_even.list[cache_entry_index - lookup];
                } else {
                    scan_odd.list[cache_entry_index] = scan_even.list[cache_entry_index];
                }
            }
            even_pass = !even_pass;
        }
        storageBarrier();
    }

    // rearrange lru entries
    if (!OUT_OF_BOUNDS) {
        // determine number of used entries
        let last_entry_index = num_entries - 1u;
        let num_used_total = get_max_count(last_entry_index);
        if (cache_entry_index == last_entry_index) {
            num_used_entries.num = num_used_total;
        }

        // write LRU index to new location
        let count = get_max_count(cache_entry_index);
        var index = cache_entry_index + num_used_total - count;
        if (used) {
            index = count - 1u;
        }
        lru_cache.list[index] = lru_entry;
    }
}

fn get_max_count(index: u32) -> u32 {
    return max(scan_even.list[index], scan_odd.list[index]);
}