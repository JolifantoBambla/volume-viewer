@include(gpu_list)

struct NumUsedEntries {
    num: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> lru_cache: ListU32;
@group(0) @binding(1) var<storage, read_write> num_used_entries: NumUsedEntries;
@group(0) @binding(2) var<storage, read_write> offsets: ListU32;
@group(0) @binding(3) var<storage, read_write> used_buffer: ListU32;

const WORKGROUP_SIZE: u32 = 256;

fn get_num_entries() -> u32 {
    return arrayLength(&lru_cache.list);
}

fn is_out_of_bounds(global_id: u32) -> bool {
    return global_id >= get_num_entries();
}


