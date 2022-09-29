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
@group(1) @binding(0) var<storage, read_write> lru_temp: ListU32;
@group(1) @binding(1) var<storage, read_write> scan: ListU32;

@compute
@workgroup_size(128, 1, 1)
fn main(@builtin(local_invocation_index) local_id: u32,
        @builtin(global_invocation_id) global_invocation_id: uint3,
        @builtin(workgroup_id) workgroup_grid_id: uint3,
        @builtin(num_workgroups) num_workgroups_in_grid: uint3) {

    // one workgroup only!
    let num_entries = arrayLength(&lru_cache.list);
    let num_used_total = num_used_entries.num;
    let block_size = 128u;
    let num_blocks = u32(ceil(f32(num_entries) / f32(block_size)));

    for (var block = 0u; block < num_blocks; block += 1u) {
        let block_offset = block_size * block;
        let global_id = block_offset + local_id;

        let OUT_OF_BOUNDS = global_id >= num_entries;

        if (block != 0u && !OUT_OF_BOUNDS) {
            scan.list[global_id] += scan.list[block_offset - 1u];
        }
        storageBarrier();

        if (!OUT_OF_BOUNDS) {
            let lru_entry = lru_cache.list[global_id];
            let usage_index = int3(index_to_subscript(lru_entry, buffer_size));
            let used = timestamp.now == u32(textureLoad(usage_buffer, usage_index, 0).r);

            let count = scan.list[global_id];
            var index = global_id + num_used_total - count;
            if (used) {
                index = count - 1u;
            }
            lru_temp.list[index] = lru_entry;
        }
        storageBarrier();
    }
}
