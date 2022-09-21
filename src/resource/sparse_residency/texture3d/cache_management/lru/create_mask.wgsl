@include(gpu_list)
@include(timestamp)
@include(type_alias)
@include(util)

@group(0) @binding(0) var usage_buffer: texture_3d<u32>;
@group(0) @binding(1) var<uniform> timestamp: Timestamp;
@group(0) @binding(2) var<storage, read_write> mask: ListU32;

@compute
@workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    let buffer_size = uint3(textureDimensions(usage_buffer));
    if any(buffer_size < global_id) {
        return;
    }
    let cache_entry_index = subscript_to_index(global_id, buffer_size);
    mask[cache_entry_index] = u32(timestamp.now == u32(textureLoad(usage_buffer, int3(global_id), 0).r));
}