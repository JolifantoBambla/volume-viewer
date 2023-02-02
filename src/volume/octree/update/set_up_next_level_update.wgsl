@include(dispatch_indirect)
@include(page_table)
@include(volume_subdivision)
@include(volume_subdivision_util)

// (read-only) global data bind group
@group(0) @binding(0) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(0) @binding(1) var<uniform> page_directory_meta: PageDirectoryMeta;

// (read-only) cache update bind group
@group(1) @binding(0) var<uniform> subdivision_index: u32;
// masks
@group(1) @binding(1) var<storage, read_write> node_helper_buffer_b: array<u32>;

// maxima
@group(2) @binding(0) var<storage, read_write> next_level_update_indirect: DispatchWorkgroupsIndirect;
@group(2) @binding(1) var<storage, read_write> num_nodes_next_level: atomic<u32>;
@group(2) @binding(2) var<storage, read_write> node_helper_buffer_a: array<u32>;

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let num_channels = page_directory_meta.max_channels;

    let multichannel_local_node_index = global_invocation_id.x;
    let num_nodes = subdivision_idx_num_nodes_in_subdivision(subdivision_index) * num_channels;
    if (multichannel_local_node_index >= num_nodes) {
        return;
    }

    let update_node = bool(node_helper_buffer_b[multichannel_local_node_index]);
    if (update_node) {
        let index = atomicAdd(&num_nodes_next_level, 1);
        atomicMax(&next_level_update_indirect.workgroup_count_x, max(index / 64, 1));
        node_helper_buffer_a[index] = multichannel_local_node_index;
    }
}
