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
@group(1) @binding(0) var<storage, read_write> scan_even: ListU32;
@group(1) @binding(1) var<storage, read_write> scan_odd: ListU32;

@compute
@workgroup_size(128, 1, 1)
fn group_wise_scan(@builtin(local_invocation_index) local_id: u32,
        @builtin(global_invocation_id) global_invocation_id: uint3,
        @builtin(num_workgroups) num_workgroups_in_grid: uint3) {
    let num_workgroups = num_workgroups_in_grid.x;
    let global_id = global_invocation_id.x;
    let buffer_size = uint3(textureDimensions(usage_buffer));
    let num_entries = buffer_size.x * buffer_size.y * buffer_size.z;
    let workgroup_size = u32(ceil(f32(num_entries) / f32(num_workgroups)));

    // note: we can't return here because then storageBarrier calls would be outside of uniform control flow, so what we
    // do instead is that we wrap all operations in if-else blocks checking for `OUT_OF_BOUNDS`
    let OUT_OF_BOUNDS = global_id >= num_entries;

    // initialize number of used entries & scan at own index
    var lru_entry = 0u;
    var used = false;
    if (!OUT_OF_BOUNDS) {
        lru_entry = lru_cache.list[global_id];
        let usage_index = int3(index_to_subscript(lru_entry, buffer_size));
        used = timestamp.now == u32(textureLoad(usage_buffer, usage_index, 0).r);
        scan_odd.list[global_id] = u32(used);
        if (used) {
            atomicAdd(&num_used_entries.num, 1u);
        }
    }

    storageBarrier();

    // scan within the workgroup
    var even_pass = true;
    for (var lookup = 1u; lookup <= workgroup_size - 1; lookup *= 2u) {
        if (!OUT_OF_BOUNDS) {
            if (even_pass) {
                if (local_id >= lookup) {
                    scan_even.list[global_id] = scan_odd.list[global_id] + scan_odd.list[global_id - lookup];
                } else {
                    scan_even.list[global_id] = scan_odd.list[global_id];
                }
            } else {
                if (local_id >= lookup) {
                    scan_odd.list[global_id] = scan_even.list[global_id] + scan_even.list[global_id - lookup];
                } else {
                    scan_odd.list[global_id] = scan_even.list[global_id];
                }
            }
            even_pass = !even_pass;
        }
        storageBarrier();
    }

    // ensure both scan arrays store the same data (this is redundant, but time is short and I don't want to add logic
    // to choose the correct array in the other pass)
    if (!OUT_OF_BOUNDS) {
        if (even_pass) {
            scan_even.list[global_id] = scan_odd.list[global_id];
        } else {
            scan_odd.list[global_id] = scan_even.list[global_id];
        }
    }
}

@compute
@workgroup_size(128, 1, 1)
fn accumulate_and_update_lru(@builtin(local_invocation_index) local_id: u32,
        @builtin(global_invocation_id) global_invocation_id: uint3,
        @builtin(workgroup_id) workgroup_grid_id: uint3,
        @builtin(num_workgroups) num_workgroups_in_grid: uint3) {

    // one workgroup only!
    let num_entries = arrayLength(&lru_cache.list);
    let num_used_total = atomicLoad(&num_used_entries.num);
    let block_size = 128u;
    let num_blocks = u32(ceil(f32(num_entries) / f32(block_size)));
    let buffer_size = uint3(textureDimensions(usage_buffer));

    for (var block = 0u; block < num_blocks; block += 1u) {
        let block_offset = block_size * block;
        let global_id = block_offset + local_id;

        let OUT_OF_BOUNDS = global_id >= num_entries;

        if (block != 0u && !OUT_OF_BOUNDS) {
            scan_even.list[global_id] += scan_even.list[block_offset - 1u];
        }
        storageBarrier();

        if (!OUT_OF_BOUNDS) {
            let lru_entry = lru_cache.list[global_id];
            let usage_index = int3(index_to_subscript(lru_entry, buffer_size));
            let used = timestamp.now == u32(textureLoad(usage_buffer, usage_index, 0).r);

            let count = scan_even.list[global_id];
            var index = global_id + num_used_total - count;
            if (used) {
                index = count - 1u;
            }
            scan_odd.list[index] = lru_entry;
        }
        storageBarrier();
    }
}

