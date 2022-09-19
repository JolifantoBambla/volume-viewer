@include(type_alias)

struct Ray {
    origin: float3,
    direction: float3,
    tmax: f32,
}

// todo: construction cost is probably not super high and the compiler will optimize this, but maybe use refs instead
fn transform_ray(ray: Ray, transform: float4x4) -> Ray {
    return Ray(
        (transform * float4(ray.origin, 1.)).xyz,
        (transform * float4(ray.direction, 0.)).xyz,
        ray.tmax
    );
}

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.direction * t;
}

struct Intersection {
    hit: bool,
    t_min: f32,
    t_max: f32,
}
