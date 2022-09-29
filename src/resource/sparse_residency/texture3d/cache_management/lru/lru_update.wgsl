@include(gpu_list)
@include(timestamp)
@include(type_alias)
@include(util)

struct NumUsedEntries {
    num: atomic<u32>,
}

let STATE_NOT_READY = 0u;
let STATE_READY     = 1u;

struct WorkgroupState {
    list: array<atomic<u32>>,
}

@group(0) @binding(0) var usage_buffer: texture_3d<u32>;
@group(0) @binding(1) var<uniform> timestamp: Timestamp;
@group(0) @binding(2) var<storage, read_write> lru_cache: ListU32;
@group(0) @binding(3) var<storage, read_write> num_used_entries: NumUsedEntries;

// these resources are never used outside of this pass
@group(1) @binding(0) var<storage, read_write> scan_even: ListU32;
@group(1) @binding(1) var<storage, read_write> scan_odd: ListU32;
@group(1) @binding(2) var<storage, read_write> group_state: WorkgroupState;

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_index) local_id: u32,
        @builtin(global_invocation_id) global_invocation_id: uint3,
        @builtin(workgroup_id) workgroup_grid_id: uint3,
        @builtin(num_workgroups) num_workgroups_in_grid: uint3) {
    //if (local_id == 0u) {
    //    atomicStore(&group_state.list[workgroup_id], STATE_NOT_READY);
    //}
    let num_workgroups = num_workgroups_in_grid.x;
    let workgroup_id = workgroup_grid_id.x;
    let global_id = global_invocation_id.x;
    let buffer_size = uint3(textureDimensions(usage_buffer));
    let num_entries = buffer_size.x * buffer_size.y * buffer_size.z;
    let workgroup_size = u32(ceil(f32(num_entries) / f32(num_workgroups)));

    // note: we can't return here because then storageBarrier calls would be outside of uniform control flow, so what we
    // do instead is that we wrap all operations in if-else blocks checking for `OUT_OF_BOUNDS`
    let OUT_OF_BOUNDS = global_id >= num_entries;

    // initialize number of used entries
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

    // after this point, all threads in the workgroup are guaranteed to have used & scan_odd initialized
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


    // now all threads in a workgroup have completed their local scan


    // next pass is num_workgroups / 128
    // each thread accumulates


    if (workgroup_id == 0u) {
        if (local_id == 0u) {
            atomicStore(&group_state.list[workgroup_id], STATE_READY);
        }
    } else {
        for (;!OUT_OF_BOUNDS;) {
            let previous_group_state = atomicLoad(&group_state.list[workgroup_id - 1]);
            if (previous_group_state == STATE_READY) {
                let previous_aggregate_id = global_id - local_id - 1u;
                if (even_pass) {
                    scan_even.list[global_id] += scan_odd.list[previous_aggregate_id];
                } else {
                    scan_odd.list[global_id] += scan_even.list[previous_aggregate_id];
                }
                if (local_id == workgroup_size - 1u) {
                    atomicStore(&group_state.list[workgroup_id], STATE_READY);
                }
                break;
            }
        }
    }
    // now each thread in the workgroup has the complete inclusive scan left to its index

    // ensure all workgroups are done
    for (;atomicLoad(&group_state.list[num_workgroups - 1u]) != STATE_READY;) {}
    let num_used_total = atomicLoad(&num_used_entries.num);

    // rearrange lru entries
    if (!OUT_OF_BOUNDS) {
        // write LRU index to new location
        let count = get_max_count(global_id);
        var index = global_id + num_used_total - count;
        if (used) {
            index = count - 1u;
        }
        lru_cache.list[index] = lru_entry;
    }

    storageBarrier();

    scan_odd.list[global_id] = global_id;
}

fn get_max_count(index: u32) -> u32 {
    return max(scan_even.list[index], scan_odd.list[index]);
}