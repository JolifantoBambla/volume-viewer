// Prefix sum
// Source: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// TODO implement "avoiding bank conflicts"

 // Input buffer
@group(0) @binding(0) var<storage, read_write> data_in_out: array<u32>;

// Workgroup buffer, with the sum of all input values of a workgroup
@group(0) @binding(1) var<storage, read_write> wgSum: array<u32>;

const WORKGROUP_SIZE = 256u;
const TEMP_SIZE = WORKGROUP_SIZE * 2;

// Local (workgroup) buffers
// Twice the workgroup size because each thread will take care of 2 elements in the buffer
var<workgroup> wg_sum: array<u32, TEMP_SIZE>;

// MAIN CODE -----------------------------------------------------------------------------------------------------------

fn prepare_workgroup_memory(
    g_address1: u32, g_address2: u32,
    l_address1: u32, l_address2: u32,
    num_elements: u32,
) -> u32 {
    // Load input into workgroup memory
    if (g_address2 < num_elements) {
        let v1 = data_in_out[g_address1];
        let v2 = data_in_out[g_address2];
        wg_sum[l_address1] = v1;
        wg_sum[l_address2] = v2;
    // Or write zero when out of bounds
    } else if (g_address1 < num_elements) {
        let v1 = data_in_out[g_address1];
        wg_sum[l_address1] = v1;
        wg_sum[l_address2] = 0u;
    } else {
        wg_sum[l_address1] = 0u;
        wg_sum[l_address2] = 0u;
    }
    return wg_sum[l_address2];
}

fn up_sweep(local_thread_id: u32, l_address1: u32, l_address2: u32, offset: ptr<function, u32, read_write>) {
    // Build sum in place UP the tree
    for (var d = TEMP_SIZE >> 1u; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (local_thread_id < d) {
            let ai = *offset * (l_address1 + 1u) - 1u;
            let bi = *offset * (l_address2 + 1u) - 1u;
            wg_sum[bi] = wg_sum[bi] + wg_sum[ai];     // Increment
        }
        *offset = *offset * 2u;
    }
}

fn down_sweep(local_thread_id: u32, l_address1: u32, l_address2: u32, offset: ptr<function, u32, read_write>) {
    // Traverse DOWN tree and build scan
    for (var d = 1u; d < TEMP_SIZE; d = d * 2u) {
        *offset = *offset >> 1u;
        workgroupBarrier();
        if (local_thread_id < d) {
            let ai = *offset * (l_address1 + 1u) - 1u;
            let bi = *offset * (l_address2 + 1u) - 1u;
            let ta = wg_sum[ai];  // Swap
            let tb = wg_sum[bi];  // Swap
            wg_sum[ai] = tb;      // Swap
            wg_sum[bi] = tb + ta; // Swap with addition!
        }
    }
}

fn write_results(
    workgroup_id: vec3<u32>,
    g_address1: u32, g_address2: u32,
    l_address1: u32, l_address2: u32,
    num_elements: u32,
    last_element: u32,
) {
    // Write results to output buffer (or don't; when out of bounds)
    workgroupBarrier();
    if (g_address2 < num_elements) {
        data_in_out[g_address1] = wg_sum[l_address1];
        data_in_out[g_address2] = wg_sum[l_address2];
    } else if (g_address1 < num_elements) {
        data_in_out[g_address1] = wg_sum[l_address1];
    }

    if (l_address2 == TEMP_SIZE - 1u) {
        wgSum[workgroup_id.x] = wg_sum[l_address2] + last_element;
    }
}

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let g_address1: u32 = 2u * global_id.x; // Global address #1
    let g_address2: u32 = g_address1 + 1u;        // Global address #2

    let local_thread_id: u32 = local_id.x;
    let l_address1: u32 = 2u * local_thread_id;        // Local workgroup ID #1
    let l_address2: u32 = l_address1 + 1u;       // Local workgroup ID #2

    let num_elements: u32 = arrayLength(&data_in_out);        // Size of the buffer
    var offset: u32 = 1u;

    let last_element = prepare_workgroup_memory(
        g_address1, g_address2,
        l_address1, l_address2,
        num_elements,
    );

    up_sweep(
        local_thread_id,
        l_address1, l_address2,
        &offset
    );

    // Clear the last element
    if (local_thread_id == 0u) {
        wg_sum[TEMP_SIZE - 1u] = 0u;
    }

    down_sweep(
        local_thread_id,
        l_address1, l_address2,
        &offset
    );

    write_results(
        workgroup_id,
        g_address1, g_address2,
        l_address1, l_address2,
        num_elements,
        last_element,
    );
}
