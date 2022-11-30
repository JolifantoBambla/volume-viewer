@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> addend: array<u32>;

const WORKGROUP_SIZE: u32 = 256;
const WORKGROUP_SIZE_DOUBLED: u32 = WORKGROUP_SIZE * 2;

@compute
@workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let num_elements: u32 = arrayLength(&data);
    let i: u32 = global_id.x;
    // Guard against out-of-bounds work group sizes
    if (i >= num_elements) {
        return;
    }
    let j: u32 = u32(floor(f32(i) / f32(WORKGROUP_SIZE_DOUBLED))); // Addend index
    data[i] = data[i] + addend[j];
}
