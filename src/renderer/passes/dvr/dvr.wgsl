//

// Type aliases (because writing vec3<f32> is annoying)
// todo: maybe use vec3f and stuff instead?
type int2 = vec2<i32>;
type int3 = vec3<i32>;
type int4 = vec4<i32>;
type uint2 = vec2<u32>;
type uint3 = vec3<u32>;
type uint4 = vec4<u32>;
type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;
type float3x3 = mat3x3<f32>;
type float4x4 = mat4x4<f32>;

// todo: come up with a better name...
struct Uniforms {
    world_to_object: float4x4,
    screen_to_camera: float4x4,
    camera_to_world: float4x4,
    volume_color: float4,
}

@group(0)
@binding(0)
var volume_data: texture_3d<f32>;

@group(0)
@binding(1)
var volume_sampler: sampler;

@group(0)
@binding(2)
var result: texture_storage_2d<rgba8unorm, write>;

@group(0)
@binding(3)
var<uniform> uniforms: Uniforms;


// no idea how infinity is written out in WGSL
//const positive_infinity: f32 =  1. / 0.;
//const negative_infinity: f32 = -1. / 0.;

// todo: use const stuff as soon as chromium supports it
fn positive_infinity() -> f32 {
    return 1. / 0.;
}

fn sample(x: vec3<f32>) -> f32 {
    return f32(textureSampleLevel(volume_data, volume_sampler, x, 0.).x);
}


/// An Axis Aligned Bounding Box (AABB)
struct AABB {
    @size(16) min: float3,
    @size(16) max: float3,
}

struct Volume {
    bounds: AABB,
    max_value: f32,
}

struct Ray {
    origin: float3,
    direction: float3,
    tmax: f32,
}

fn transform_ray(ray: Ray, transform: float4x4) -> Ray {
    return Ray(
        (float4(ray.origin, 1.) * transform).xyz,
        (float4(ray.direction, 0.) * transform).xyz,
        ray.tmax
    );
}

// todo: add camera models
fn generate_ray(pixel_position: float2, screen_resolution: float2, to_camera: float4x4, to_world: float4x4, to_object: float4x4) -> Ray {
    let camera_point = (float4(pixel_position + 0.5, -10000., 1.) * to_camera).xyz;

    // persp
    //let origin = float3(0.);
    //let direction = normalize(camera_point);

    // ortho
    let origin = camera_point;
    let direction = float3(0., 0., 1.);

    return transform_ray(
        //transform_ray(
            Ray(
                origin,
                direction,
                positive_infinity()
            ),
            to_world
        //),
        //to_object
    );
}

struct Intersection {
    hit: bool,
    t_min: f32,
    t_max: f32,
}

// x: true/false, y: t_min, z: t_max
fn intersect_box(ray: Ray, bounds: AABB) -> float3 {
    let inv_ray_dir = 1. / ray.direction;
    let t_near = (bounds.min - ray.origin) * inv_ray_dir;
    let t_far = (bounds.max - ray.origin) * inv_ray_dir;
    let t0 = max(float3(0.), min(t_near, t_far));
    let t1 = min(float3(ray.tmax), max(t_near, t_far));
    if any(t0 > t1) {
        return float3(-1.);
    }
    return float3(
        1.,
        max(t0.x, max(t0.y, t0.z)),
        min(t1.x, max(t1.y, t1.z))
    );
}

// x: true/false, y: t_min, z: t_max
fn intersect(ray: Ray, bounds: AABB) -> Intersection {
    var t0 = 0.;
    var t1 = ray.tmax;
    for (var i: i32 = 0; i < 3; i += 1) {
        let inv_ray_dir = 1. / ray.direction[i];
        var t_near = (bounds.min[i] - ray.origin[i]) * inv_ray_dir;
        var t_far = (bounds.max[i] - ray.origin[i]) * inv_ray_dir;
        if t_near > t_far {
            let helper = t_near;
            t_near = t_far;
            t_far = helper;
        }
        t0 = max(t_near, t0);
        t1 = min(t_far, t1);
        if t0 > t1 {
            return Intersection();
        }
    }
    return Intersection(true, t0, t1);
}

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.direction * t;
}

fn max_steps_for_size(size: vec3<f32>) -> i32 {
    return i32(ceil((size.x * size.y * size.z) * sqrt(3.)));
}

// todo: const (not supported yet)
let relative_step_size: f32 = 1.;

// todo: customizable background
let background_color = float3(0.2);


fn hit_sphere_at_origin(ray: Ray, radius: f32) -> f32 {
    return hit_sphere(ray, float3(0.), radius);
}

fn add_sphere(ray: Ray, pixel: int2, center: float3, radius: f32, color: float3) {
    let hit = hit_sphere(ray, center, 0.5);
    if hit > 0. {
        let normal = normalize(ray_at(ray, hit) - center);
        let light_dir = normalize(float3(1.));
        let lighting = dot(normal, light_dir) * color;
        textureStore(result, pixel, float4(lighting, 1.));
    }
}

fn hit_sphere(ray: Ray, center: float3, radius: f32) -> f32 {
    let origin = ray.origin - center;
    let a = dot(ray.direction, ray.direction);
    let b = 2. * dot(ray.direction, origin);
    let c = dot(origin, origin) - (radius * radius);
    let discriminant = b * b - 4. * a * c;
    if discriminant < 0. { return 0.; }
    var root_discriminant = sqrt(discriminant);
    if b < 0. {
        root_discriminant *= -1.;
    }
    let q = -0.5 * (b + root_discriminant);
    var t0 = q / a;
    var t1 = c / q;
    if t0 > t1 {
        let helper = t0;
        t0 = t1;
        t1 = helper;
    }
    if t0 > ray.tmax || t1 <= 0. { return 0.; }
    if t0 < 0. && t1 > ray.tmax { return 0.; }
    if t0 < 0. { return t1; } else { return t0; }
}

fn add_camera_debug_spheres(ray: Ray, pixel: int2) {
    add_sphere(ray, pixel, float3(0.), 0.5, float3(1.));
    add_sphere(ray, pixel, float3(-1., -1., 0.), 0.5, float3(1., 0., 0.));
    add_sphere(ray, pixel, float3(1., 1., 0.), 0.5, float3(0., 1., 0.));
    add_sphere(ray, pixel, float3(-1., 1., 0.), 0.5, float3(0., 0., 1.));
    add_sphere(ray, pixel, float3(1., -1., 0.), 0.5, float3(0.5, 0.5, 0.));
}

// supported builtins:
// - local_invocation_id    : vec3<u32>
// - local_invocation_index : u32
// - global_invocation_id   : vec3<u32>
// - workgroup_id           : vec3<u32>
// - num_workgroups         : vec3<u32>
@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // todo:
    //  - generate ray
    //  - transform ray to camera's view space
    //  - set up volume bounds
    //  - step through volume until boundary is hit
    // -> required uniforms: camera.view, volume (3d texture + bounds)

    let window_size = uint2(textureDimensions(result));

    // terminate thread if it's outside the window bounds
    if window_size.x < global_id.x || window_size.y < global_id.y {
        return;
    }

    // todo: maybe just construct camera frame in shader
    let pixel = vec2<i32>(
        i32(global_id.x),
        // invert y
        //i32(window_size.y - global_id.y));
        i32(global_id.y));

    // initialize background
    textureStore(result, pixel, float4(background_color, 1.));

    // todo: clean up ray generation and transform to volume's space s.t. volume shape is preserved and sampling still works
    var ray = generate_ray(
        float2(pixel) - float2(window_size) / 2.,
        float2(window_size),
        uniforms.screen_to_camera,
        uniforms.camera_to_world,
        uniforms.world_to_object
    );

    let volume = Volume (
        AABB (float3(0.), float3(1.)),
        1.
    );

    let intersection_record = intersect(ray, volume.bounds);

    // early out if we missed the volume
    if !intersection_record.hit {
        return;
    }

    let start = ray_at(ray, intersection_record.t_min);
    let end = ray_at(ray, intersection_record.t_max);

    // todo: remove this
    let volume_scale = float3(textureDimensions(volume_data));

    let distance = distance(start * volume_scale, end * volume_scale);
    let num_steps = i32(distance / relative_step_size + 0.5);

    // early out if we there is not a single sample within the volume
    if num_steps < 1 {
        return;
    }

    let step = (end - start) / f32(num_steps);

    //let value = float3(sample(start));
    //textureStore(result, pixel, float4(value, 1.));

    let color = dvr(start, step, num_steps, ray);
    if color.a > 0.1 {
        textureStore(result, pixel, color);
    }

    //add_camera_debug_spheres(ray, pixel);
}

fn dvr(start: float3, step: float3, num_steps: i32, ray: Ray) -> float4 {
    var color = float4(0.);

    var sample_location = start;
    for (var i: i32 = 0; i < num_steps; i += 1) {
        let value = sample(sample_location);

        let dt_scale = 1.;

        // todo: add thresholding to filter out noise
        if value > 0.1 {
            var lighting = compute_lighting(value, sample_location, step, ray.direction);
            lighting.a = 1. - pow(1. - lighting.a, dt_scale);
            color += float4(lighting.rgb, 1.) * ((1. - color.a) * lighting.a);
        }

        // early out for high alpha values
        if color.a > 0.95 {
            break;
        }

        sample_location += step;
    }

    return color;
}

fn central_difference(x: float3, step: float3) -> float3 {
    var central_difference = float3();
    for (var i = 0; i < 3; i += 1) {
        let h = step[i];
        var offset = float3();
        offset[i] = h;
        central_difference[i] = sample(x - offset) - sample(x + offset) / (2. * h);
    }
    return central_difference;
}

/// Computes the normal at point x from the central difference.
fn compute_volume_normal(x: float3, step: float3, view: float3) -> float3 {
    let normal = normalize(central_difference(x, step));
    // flip normal towards viewer if necessary
    let front_or_back = 2. * f32(dot(normal, view) > 0.) - 1.;
    return front_or_back * normal;
}

fn compute_lighting(value: f32, x: float3, step: float3, view_direction: float3) -> float4 {
    // Calculate color by incorporating lighting

    // View direction
    let view = normalize(view_direction);
    // calculate normal vector from gradient
    let normal = compute_volume_normal(x, step, view);

    // Init colors
    var ambient_color  = float3(0.1, 0.1, 0.1);
    var diffuse_color  = float3(0.0, 0.0, 0.0);
    var specular_color = float3(0.0, 0.0, 0.0);
    let shininess = 40.;

    // note: could allow multiple lights
    for (var i = 0; i < 1; i += 1) {
        // Get light direction (make sure to prevent zero devision)
        var light_direction = view;	//lightDirs[i];

        // Calculate lighting properties
        let diffuse = clamp(dot(normal, light_direction), 0., 1.);
        let halfway = normalize(light_direction + view);
        let specular = pow(max(dot(halfway, normal), 0.), shininess);

        // Calculate colors
        ambient_color  += ambient_color;
        diffuse_color  += diffuse;
        specular_color += specular * specular_color;
    }

    // Calculate final color by componing different components
    return float4(
        apply_colormap(value) * (ambient_color + diffuse_color) + specular_color,
        value
    );
}

fn apply_colormap(value: f32) -> float3{
    let u_clim = float2(0., 0.5);
    return uniforms.volume_color.rgb * (value - u_clim[0]) / (u_clim[1] - u_clim[0]);
}
