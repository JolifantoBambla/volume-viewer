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
    if (any(buffer_size < global_id)) {
        return;
    }

    let num_entries = buffer_size.x * buffer_size.y * buffer_size.z;
    let cache_entry_index = subscript_to_index(global_id, buffer_size);

    // initialize mask and number of used entries
    let used = u32(timestamp.now == u32(textureLoad(usage_buffer, int3(global_id), 0).r));
    if (used) {
        atomicAdd(&num_used_entries.num);
    }
    mask[cache_entry_index] = used;
    scan_odd[cache_entry_index] = used;

    storageBarrier();

    // determine number of used entries to the left of the current entry including the current entry itself
    var even_pass = true;
    for (var lookup_power_of_2 = 0u; lookup_power_of_2 <= num_entries; lookup_power_of_2 += 1u) {
        let lookup = 1u << lookup_power_of_2;
        if (even_pass) {
            if (cache_entry_index <= lookup) {
                scan_even[cache_entry_index] = scan_odd[cache_entry_index] + scan_odd[cache_entry_index - lookup];
            } else {
                scan_even[cache_entry_index] = scan_odd[cache_entry_index];
            }
        } else {
            if (cache_entry_index <= lookup) {
                scan_odd[cache_entry_index] = scan_even[cache_entry_index] + scan_even[cache_entry_index - lookup];
            } else {
                scan_odd[cache_entry_index] = scan_even[cache_entry_index];
            }
        }
        even_pass = !even_pass;
        storageBarrier();
    }

    // rearrange lru entries into temporary storage based on usage
    var index = 0u;
    if (even_pass) {
        if (mask[cache_entry_index]) {
            index = scan_odd[cache_entry_index] - 1u;
        } else {
            index = cache_entry_index + (num_used_entries.num - scan_odd[cache_entry_index]);
        }
        scan_even[cache_entry_index] = lru_cache[index];
    } else {
        if (mask[cache_entry_index]) {
            index = scan_even[cache_entry_index] - 1u;
        } else {
            index = cache_entry_index + (num_used_entries.num - scan_even[cache_entry_index]);
        }
        scan_odd[cache_entry_index] = lru_cache[index];
    }

    storageBarrier();

    // copy result back to lru_cache
    if (even_pass) {
        lru_cache[cache_entry_index] = scan_even[cache_entry_index];
    } else {
        lru_cache[cache_entry_index] = scan_odd[cache_entry_index];
    }
}