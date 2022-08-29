@include(constant)
@include(page_table)
@include(ray)
@include(type_alias)

struct VoxelLineState {
    // the position along the ray where it enters the current brick
    entry: float3,

    // the position along the ray where it exits the current brick
    exit: float3,

    // the current brick address along a VoxelLine's line
    brick: int3,

    // the ray coordinate t at which the ray enters the current brick
    t_min: f32,

    // the ray coordinate t at which the ray crosses the next grid plane (per dimension)
    t_max: float3,

    // the dimension along which the next step is taken
    next_step_dimension: u32,

    // the number of steps that where taken along the VoxelLine's line per dimension
    steps: uint3,
}

/// A line voxelization for a ray intersecting a grid in the unit cube (x in [0.0, 1.0]^3)
struct VoxelLine {
    // never changes

    grid_min: uint3,
    grid_max: uint3,

    // the step direction in each dimension in bricks (x in [-1, 0, 1])
    brick_step: int3,

    // the last brick the ray travels through before leaving the volume
    last_brick: int3,

    // the step size along the ray for the ray to cross a grid plane (per dimension)
    t_delta: float3,

    // the ray coordinate t at which the ray left the first voxel along the line
    first_t_max: float3,

    // the current state during line traversal
    // it starts at the first voxel along the line and is updated every time `advance` is called
    state: VoxelLineState,
}

fn create_voxel_line(start: float3, stop: float3, t_min: f32, ray: Ray, page_table: ptr<function, PageTableMeta, read_write>) -> VoxelLine {
    let pt = *page_table;
    let grid_min = pt.page_table_offset;
    let grid_max = grid_min + pt.page_table_extent;

    let brick_step = int3(sign(ray.direction));
    let current_position = start;

    let volume_to_padded = compute_volume_to_padded(page_table);
    let current_brick = int3(compute_page_address(page_table, start * volume_to_padded));
    let last_brick = int3(compute_page_address(page_table, stop * volume_to_padded));

    let normalized_brick_size = 1. / float3(pt.page_table_extent);
    let t_delta = normalized_brick_size / (ray.direction * float3(brick_step));

    let previous_brick = current_brick - clamp(brick_step, int3(-1), int3(0)) * -1;
    let t_max = t_min + (float3(previous_brick) * normalized_brick_size - current_position) / ray.direction;

    let next_step_dimension = min_dimension(t_max);
    let current_exit = clamp(ray_at(ray, t_max[next_step_dimension] - EPSILON), float3(), float3(1.));

    let state = VoxelLineState(
        current_position,
        current_exit,
        current_brick,
        t_min,
        t_max,
        next_step_dimension,
        uint3()
    );

    return VoxelLine(
        grid_min,
        grid_max,
        brick_step,
        last_brick,
        t_delta,
        t_max,
        state
    );
}

fn advance(voxel_line: ptr<function, VoxelLine, read_write>, ray: Ray) {
    let vl = *voxel_line;
    let d = vl.state.next_step_dimension;

    (*voxel_line).state.steps[d] += 1u;
    (*voxel_line).state.brick[d] += vl.brick_step[d];
    (*voxel_line).state.t_min = vl.state.t_max[d];
    (*voxel_line).state.t_max[d] = vl.first_t_max[d] + f32(vl.state.steps[d]) * vl.t_delta[d];

    (*voxel_line).state.next_step_dimension = min_dimension(vl.state.t_max);

    (*voxel_line).state.entry = clamp(ray_at(ray, vl.state.t_min + EPSILON), float3(), float3(1.));
    (*voxel_line).state.exit  = clamp(ray_at(ray, vl.state.t_max[vl.state.next_step_dimension] - EPSILON), float3(), float3(1.));
}

fn in_grid(voxel_line: ptr<function, VoxelLine, read_write>) -> bool {
    let vl = *voxel_line;
    let brick = uint3(vl.state.brick);
    return all(brick >= vl.grid_min) && all(brick <= vl.grid_max);
}