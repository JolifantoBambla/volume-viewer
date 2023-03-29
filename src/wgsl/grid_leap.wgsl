@include(ray)

// todo: there is a better name for sure ;)
struct GridRay {
    inv_ray_dir: vec3<f32>,
    ray_signs: vec3<f32>,
    grid_step: vec3<i32>,
    t_max: f32,
}

fn make_grid_ray(ray: Ray, t_max: f32) -> GridRay {
    let inv_ray_dir = 1. / ray.direction;
    let ray_signs = sign(ray.direction);
    let grid_step = vec3<i32>(saturate(ray_signs));
    return GridRay (
        inv_ray_dir,
        ray_signs,
        grid_step,
        t_max
    );
}

fn compute_jump_distance(position: vec3<f32>, next_axes: vec3<f32>, dt: f32, grid_ray: GridRay) -> f32 {
    // for each axis, we compute the distance to the next axis intersection
    var t_jump = grid_ray.t_max;
    for (var i: u32 = 0u; i < 3u; i += 1u) {
        if (grid_ray.ray_signs[i] != 0) { // prevent division by zero
            let jump = (next_axes[i] - position[i]) * grid_ray.inv_ray_dir[i];
            if (jump > 0) {
                t_jump = min(t_jump, jump);
            }
        }
    }

    // We need to make sure that we jump to a sample w.r.t. the current sampling rate, so we want to jump exactly over
    // ⌊(t_jump / dt)⌋ + 1 samples.
    // We assume that this is called in a ray casting loop that advances the ray by 1 sample in its continuing block.
    // So the jump distance we want to return is ⌊(t_jump / dt)⌋ * dt.
    return dt * floor(t_jump / dt);
}