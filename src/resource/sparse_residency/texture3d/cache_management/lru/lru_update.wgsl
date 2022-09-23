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

// these resources are never used outside of this pass
@group(1) @binding(0) var<storage, read_write> mask: ListU32;
@group(1) @binding(1) var<storage, read_write> scan_even: ListU32;
@group(1) @binding(2) var<storage, read_write> scan_odd: ListU32;

@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    let buffer_size = uint3(textureDimensions(usage_buffer));

    // note: we can't return here because then storageBarrier calls would be outside of uniform control flow, so what we
    // do instead is that we wrap all operations in if-else blocks checking for `OUT_OF_BOUNDS`
    let OUT_OF_BOUNDS = any(buffer_size < global_id);

    let num_entries = buffer_size.x * buffer_size.y * buffer_size.z;
    let cache_entry_index = subscript_to_index(global_id, buffer_size);
    var lru_entry = 0u;

    // initialize mask & number of used entries
    let used = !OUT_OF_BOUNDS && timestamp.now == u32(textureLoad(usage_buffer, int3(global_id), 0).r);
    let used_storable = u32(used);
    if (used) {
        atomicAdd(&num_used_entries.num, 1u);
    }
    if (!OUT_OF_BOUNDS) {
        mask.list[cache_entry_index] = used_storable;
        scan_odd.list[cache_entry_index] = used_storable;
        lru_entry = lru_cache.list[cache_entry_index];
    }
    
    storageBarrier();

    let num_used_total = atomicLoad(&num_used_entries.num);

    // determine number of used entries to the left of the current entry including the current entry itself
    var even_pass = true;
    for (var lookup = 1u; lookup <= num_entries; lookup *= 2u) {
        if (!OUT_OF_BOUNDS) {
            if (even_pass) {
                if (cache_entry_index <= lookup) {
                    scan_even.list[cache_entry_index] = scan_odd.list[cache_entry_index] + scan_odd.list[cache_entry_index - lookup];
                } else {
                    scan_even.list[cache_entry_index] = scan_odd.list[cache_entry_index];
                }
            } else {
                if (cache_entry_index <= lookup) {
                    scan_odd.list[cache_entry_index] = scan_even.list[cache_entry_index] + scan_even.list[cache_entry_index - lookup];
                } else {
                    scan_odd.list[cache_entry_index] = scan_even.list[cache_entry_index];
                }
            }
            even_pass = !even_pass;
        }
        storageBarrier();
    }

    if (!OUT_OF_BOUNDS) {
        // rearrange lru entries
        var index = 0u;
        if (even_pass) {
            if (used) {
                index = scan_odd.list[cache_entry_index] - 1u;
            } else {
                index = cache_entry_index + (num_used_total - scan_odd.list[cache_entry_index]);
            }
        } else {
            if (used) {
                index = scan_even.list[cache_entry_index] - 1u;
            } else {
                index = cache_entry_index + (num_used_total - scan_even.list[cache_entry_index]);
            }
        }
        lru_cache.list[index] = lru_entry;
    }
}