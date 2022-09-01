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
    t_entry: f32,

    // the ray coordinate t at which the ray crosses the next grid plane (per dimension)
    t_next_crossing: float3,

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
    volume_to_padded: float3,
    inverse_brick_size: float3,

    // the step direction in each dimension in bricks (x in [-1, 0, 1])
    brick_step: int3,

    // the last brick the ray travels through before leaving the volume
    last_brick: int3,

    // the step size along the ray for the ray to cross a grid plane (per dimension)
    t_delta: float3,

    // the ray coordinate t at which the ray left the first voxel along the line
    first_t_next_crossing: float3,

    valid: u32,

    // the current state during line traversal
    // it starts at the first voxel along the line and is updated every time `advance` is called
    state: VoxelLineState,
}

/// The grid is a unit cube ([0,1]^3)
/// The ray is scaled so that the grid actually is a unit cube located at the origin
/// This means the ray's direction isn't necessarily 1.
/// This shouln't matter...
fn create_voxel_line(ray: Ray, t_entry: f32, t_exit: f32, page_table: ptr<function, PageTableMeta, read_write>) -> VoxelLine {
    let pt = *page_table;

    // todo: remove (debug)
    var valid = 0u;


    // compute basic page table properties
    let grid_min = pt.page_table_offset;
    let grid_max_index = pt.page_table_extent - uint3(1u);
    let grid_max = grid_min + grid_max_index;
    let volume_to_padded = compute_volume_to_padded(page_table);
    let normalized_brick_size = 1. / float3(pt.page_table_extent);

    // compute the entry and exit points for the grid
    let entry = clamp_to_one(ray_at(ray, t_entry) * volume_to_padded);
    let exit  = clamp_to_one(ray_at(ray, t_exit) * volume_to_padded);

    // compute first and last brick coordinates on the line
    let current_brick = int3(compute_page_address(page_table, entry));
    let last_brick    = int3(compute_page_address(page_table, exit));
    let current_local_brick_index = current_brick - int3(grid_min);

    // set up a new ray going from the current position in the grid to the point where it exits the grid w.r.t. padding
    let r = Ray(entry, ray.direction * volume_to_padded, t_exit - t_entry);

    // set up the step direction in brick indices for each dimension
    let brick_step = int3(sign(r.direction));

    // compute the step size between axis crossings per dimension
    let t_delta = normalized_brick_size / abs(r.direction);

    // compute step size to next axis crossing per dimension
    let next_axis_crossing_index = current_local_brick_index + clamp(brick_step, int3(), int3(1));
    let next_axis_crossing = float3(next_axis_crossing_index) * normalized_brick_size;
    var t_next_crossing = float3(t_exit + EPSILON);
    for (var i: u32 = 0u; i < 3u; i += 1u) {
        if (brick_step[i] != 0) {
            // todo: this is wrong somehow... how can this be < 0?
            // (a-b) / c is < 0 if...
            //  a-b < 0 && c > 0
            //  a-b > 0 && c < 0
            t_next_crossing[i] = t_entry + (next_axis_crossing[i] - r.origin[i]) / r.direction[i];

            if ((next_axis_crossing[i] - r.origin[i]) < 0. && r.direction[i] > 0.) {
                valid = 1u;
                break;
            } else if ((next_axis_crossing[i] - r.origin[i]) > 0. && r.direction[i] < 0.) {
                valid = 2u;
                break;
            }
            if (t_next_crossing[i] < t_entry) {
                valid = 3u;
                break;
            }
        }
    }

    if (all(t_next_crossing > float3(t_entry))) {
        valid = 4u;
    }

    let next_step_dimension = min_dimension(t_next_crossing);

    let current_brick_coords = float3(current_local_brick_index) * normalized_brick_size;
    let current_entry = clamp(
        (ray_at(ray, t_entry) * volume_to_padded) - current_brick_coords,
        float3(EPSILON),
        float3(normalized_brick_size - EPSILON)
    ) + current_brick_coords;
    let current_exit = clamp(
        (ray_at(ray, t_next_crossing[next_step_dimension]) * volume_to_padded) - current_brick_coords,
        float3(EPSILON),
        float3(normalized_brick_size - EPSILON)
    ) + current_brick_coords;

    let state = VoxelLineState(
        current_entry, //entry,
        current_exit,
        current_brick,
        t_entry,
        t_next_crossing,
        next_step_dimension,
        uint3()
    );

    return VoxelLine(
        grid_min,
        grid_max,
        volume_to_padded,
        normalized_brick_size,
        brick_step,
        last_brick,
        t_delta,
        t_next_crossing,
        valid,
        state
    );
}

fn compute_current_voxel_coords(voxel_line: ptr<function, VoxelLine, read_write>) -> float3 {
    let vl = *voxel_line;
    return float3(vl.state.brick - int3(vl.grid_min)) * vl.inverse_brick_size;
}

fn advance(voxel_line: ptr<function, VoxelLine, read_write>, ray: Ray) {
    let vl = *voxel_line;
    let d = vl.state.next_step_dimension;

    (*voxel_line).state.steps[d] += 1u;
    (*voxel_line).state.brick[d] += vl.brick_step[d];
    (*voxel_line).state.t_entry = vl.state.t_next_crossing[d];
    (*voxel_line).state.t_next_crossing[d] = vl.first_t_next_crossing[d] + f32(vl.state.steps[d]) * vl.t_delta[d];

    (*voxel_line).state.next_step_dimension = min_dimension(vl.state.t_next_crossing);

    let voxel_coords = compute_current_voxel_coords(voxel_line);
    (*voxel_line).state.entry = clamp(
        (ray_at(ray, vl.state.t_entry) * vl.volume_to_padded) - voxel_coords,
        float3(),
        float3(vl.inverse_brick_size)
    ) + voxel_coords;
    (*voxel_line).state.exit = clamp(
        (ray_at(ray, vl.state.t_next_crossing[vl.state.next_step_dimension]) * vl.volume_to_padded) - voxel_coords,
        float3(),
        float3(vl.inverse_brick_size)
    ) + voxel_coords;
}

fn in_grid(voxel_line: ptr<function, VoxelLine, read_write>) -> bool {
    let vl = *voxel_line;
    let brick = uint3(vl.state.brick);
    return all(brick >= vl.grid_min) && all(brick <= vl.grid_max);
}

fn is_broken(voxel_line: ptr<function, VoxelLine, read_write>) -> bool {
    let vl = *voxel_line;
    return any(vl.brick_step != int3(sign(vl.state.exit - vl.state.entry)));
}
