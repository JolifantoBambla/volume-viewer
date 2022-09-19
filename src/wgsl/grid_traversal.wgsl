@include(aabb)
@include(constant)
@include(page_table)
@include(ray)
@include(type_alias)

struct VoxelEndpoints {
    // the position along the ray where it enters the current voxel
    entry: float3,

    // the position along the ray where it exits the current voxel
    exit: float3,
}

struct TraversalState {
    // The current voxel index
    voxel: int3,

    // The ray coordinate t at which the ray enters the current voxel
    t_entry: f32,

    // The ray coordinate t at which the ray crosses the next grid plane (per dimension)
    t_next_crossing: float3,

    // The dimension along which the next step is taken
    next_step_dimension: u32,
}

/// A line voxelization for a ray intersecting a grid in the unit cube (x in [0.0, 1.0]^3)
struct GridTraversal {
    // The grid's offset in voxel coordinates
    grid_min: uint3,
    
    // The grid's end point in voxel coordinates
    grid_max: uint3,
    
    // The ratio of filled voxels to all voxels in the grid
    volume_to_padded: float3,
    
    // The inverse scale of a voxel
    inverse_voxel_scale: float3,
    
    // The ray traversing the grid.
    // Its origin is the point where it enters the grid.
    // Its direction is normalized.
    // Its tmax value is positive infinity.
    ray: Ray,

    // The step direction in each dimension in voxel coordinates (x in [-1, 0, 1])
    voxel_step: int3,

    // The step size along the ray for the ray to cross a grid plane (per dimension).
    t_delta: float3,

    // The current traversal state.
    // It starts at the first voxel along the line and is updated every time `next_voxel` is called.
    state: TraversalState,
}

/// The grid is a unit cube ([0,1]^3)
/// The ray is scaled so that the grid actually is a unit cube located at the origin
/// This means the ray's direction isn't necessarily 1.
/// This shouln't matter...
fn create_grid_traversal(ray: Ray, t_entry: f32, t_exit: f32, page_table: PageTableMeta) -> GridTraversal {
    // compute basic page table properties
    let grid_min = page_table.page_table_offset;
    let grid_max_index = page_table.page_table_extent - uint3(1u);
    let grid_max = grid_min + grid_max_index;
    let volume_to_padded = compute_volume_to_padded(page_table);
    let inverse_voxel_scale = 1. / float3(page_table.page_table_extent);

    // compute the entry and exit points for the grid
    // we move the ray a little bit forward to make sure we're in the current voxel to compute the entry pointer but
    // we don't care about the exit so much - we only use it for computing the last voxel's index
    let entry = clamp_to_one(ray_at(ray, t_entry + EPSILON) * volume_to_padded);
    let exit  = clamp_to_one(ray_at(ray, t_exit) * volume_to_padded);

    // compute first and last voxel coordinates on the line
    let current_voxel = int3(compute_page_address(page_table, entry));
    let last_voxel    = int3(compute_page_address(page_table, exit));

    // set up a new ray going from the current position in the grid to the point where it exits the grid w.r.t. padding
    let r = Ray(entry, normalize(ray.direction * volume_to_padded), positive_infinity());

    // set up the step direction in voxel indices for each dimension
    let voxel_step = int3(sign(r.direction));

    // compute the step size between axis crossings per dimension
    let t_delta = inverse_voxel_scale / abs(r.direction);

    // compute step size to next axis crossing per dimension
    let current_local_voxel_index = current_voxel - int3(grid_min);
    let next_axis_crossing_index = current_local_voxel_index + clamp(voxel_step, int3(), int3(1));
    let next_axis_crossing = float3(next_axis_crossing_index) * inverse_voxel_scale;
    var t_next_crossing = float3(r.tmax);
    for (var i: u32 = 0u; i < 3u; i += 1u) {
        if (voxel_step[i] != 0) {
            t_next_crossing[i] = (next_axis_crossing[i] - r.origin[i]) / r.direction[i];
        }
    }
    let next_step_dimension = min_dimension(t_next_crossing);

    let state = TraversalState(
        current_voxel,
        0.,
        t_next_crossing,
        next_step_dimension,
    );

    return GridTraversal(
        grid_min,
        grid_max,
        volume_to_padded,
        inverse_voxel_scale,
        r,
        voxel_step,
        t_delta,
        state
    );
}

fn _compute_current_voxel_coords(gt: GridTraversal) -> float3 {
    return float3(gt.state.voxel - int3(gt.grid_min)) * gt.inverse_voxel_scale;
}

fn _clamp_to_voxel_bounds(position: float3, voxel_bounds: AABB) -> float3 {
    return clamp(position - voxel_bounds.min, float3(), voxel_bounds.max) + voxel_bounds.min;
}

fn compute_voxel_endpoints(gt: GridTraversal) -> VoxelEndpoints {
    let voxel_bounds = AABB(
        _compute_current_voxel_coords(gt),
        gt.inverse_voxel_scale
    );
    let t_entry = gt.state.t_entry + EPSILON;
    let t_exit = max(t_entry, gt.state.t_next_crossing[gt.state.next_step_dimension] - EPSILON);
    return VoxelEndpoints(
        _clamp_to_voxel_bounds(ray_at(gt.ray, t_entry), voxel_bounds),
        _clamp_to_voxel_bounds(ray_at(gt.ray, t_exit), voxel_bounds)
    );
}

fn next_voxel(gt: ptr<function, GridTraversal, read_write>) {
    let gt_last = *gt;
    let d = gt_last.state.next_step_dimension;

    // update parameters
    (*gt).state.voxel[d] += gt_last.voxel_step[d];
    (*gt).state.t_entry = gt_last.state.t_next_crossing[d];
    (*gt).state.t_next_crossing[d] += gt_last.t_delta[d];
    (*gt).state.next_step_dimension = min_dimension((*gt).state.t_next_crossing);
}

fn in_grid(gt: GridTraversal) -> bool {
    let voxel = uint3(gt.state.voxel);
    return all(voxel >= gt.grid_min) && all(voxel <= gt.grid_max);
}
