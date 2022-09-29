@include(ray)
@include(type_alias)

struct Sphere {
    center: float3,
    radius: f32,
}

fn intersect_sphere(ray: Ray, sphere: Sphere) -> Intersection {
    let origin = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let b = 2. * dot(ray.direction, origin);
    let c = dot(origin, origin) - (sphere.radius * sphere.radius);
    let discriminant = b * b - 4. * a * c;
    if discriminant < 0. { return Intersection(); }
    var root_discriminant = sqrt(discriminant);
    if b < 0. {
        root_discriminant *= -1.;
    }
    let q = -0.5 * (b + root_discriminant);
    var t0 = q / a;
    var t1 = c / q;
    if t0 > t1 {
        swap(&t0, &t1);
    }
    if t0 > ray.tmax || t1 <= 0. { return Intersection(); }
    if t0 < 0. && t1 > ray.tmax { return Intersection(); }
    return Intersection(true, t0, t1);
}
