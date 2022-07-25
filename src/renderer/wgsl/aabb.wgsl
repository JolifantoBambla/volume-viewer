@include(ray)
@include(type_alias)
@include(util)

/// An Axis Aligned Bounding Box (AABB)
struct AABB {
    @size(16) min: float3,
    @size(16) max: float3,
}

fn intersect_aabb(ray: Ray, bounds: AABB) -> Intersection {
    var t0 = 0.;
    var t1 = ray.tmax;
    for (var i = 0; i < 3; i += 1) {
        let inv_ray_dir = 1. / ray.direction[i];
        var t_near = (bounds.min[i] - ray.origin[i]) * inv_ray_dir;
        var t_far = (bounds.max[i] - ray.origin[i]) * inv_ray_dir;
        if t_near > t_far {
            swap(&t_near, &t_far);
        }
        t0 = max(t_near, t0);
        t1 = min(t_far, t1);
        if t0 > t1 {
            return Intersection();
        }
    }
    return Intersection(true, t0, t1);
}
