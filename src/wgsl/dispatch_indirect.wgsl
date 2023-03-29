/// A helper struct for filling an indirectBuffer from a shader stage
struct DispatchWorkgroupsIndirect {
    workgroup_count_x: atomic<u32>,
    workgroup_count_y: atomic<u32>,
    workgroup_count_z: atomic<u32>,
}
