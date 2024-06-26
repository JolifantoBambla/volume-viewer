//

@include(aabb)
@include(camera)
@include(ray)
@include(sphere)
@include(transform)
@include(type_alias)

// constant values
// todo: replace let with const
// todo: maybe just remove this
let relative_step_size: f32 = 1.;
// todo: should be a uniform
let background_color = float4(float3(0.2), 1.);

fn debug(pixel: int2, color: float4) {
    textureStore(result, pixel, color);
}
fn red(pixel: int2) {
    textureStore(result, pixel, RED);
}
fn green(pixel: int2) {
    textureStore(result, pixel, GREEN);
}
fn blue(pixel: int2) {
    textureStore(result, pixel, BLUE);
}
fn white(pixel: int2) {
    textureStore(result, pixel, WHITE);
}
fn black(pixel: int2) {
    textureStore(result, pixel, BLACK);
}

struct Volume {
    transform: Transform,

    // The size of the volume.
    // Note that this field has a size of 16 bytes to avoid alignment issues.
    @size(16) dimensions: float3,
}

struct VolumeBlock {
    bounds: AABB,
    // todo: some meta information about the volume block
}

// todo: come up with a better name...
// todo: clean up those uniforms - find out what i really need and throw away the rest
struct Uniforms {
    camera: Camera,
    world_to_object: float4x4,
    object_to_world: float4x4,
    volume_color: float4,
}

// Bindings

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

// Helper stuff

fn sample_volume(x: float3) -> f32 {
    return f32(textureSampleLevel(volume_data, volume_sampler, x, 0.).x);
}

// Main stage

// supported builtins:
// - local_invocation_id    : vec3<u32>
// - local_invocation_index : u32
// - global_invocation_id   : vec3<u32>
// - workgroup_id           : vec3<u32>
// - num_workgroups         : vec3<u32>
@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let window_size = uint2(textureDimensions(result));

    // terminate thread if it's outside the window bounds
    if window_size.x < global_id.x || window_size.y < global_id.y {
        return;
    }

    let pixel = int2(global_id.xy);
    let resolution = float2(window_size);

    // initialize background
    textureStore(result, pixel, background_color);

    // todo: remove this (should come from CPU because it depends on metadata not on texture resolution)
    let volume_scale = float3(textureDimensions(volume_data));

    // generate a ray and transform it to the volume's space (i.e. where the volume is a unit cube with x in [0,1]^3)
    var ray = generate_camera_ray(uniforms.camera, float2(pixel), resolution);
    ray = transform_ray(ray, uniforms.world_to_object);

    // volume is [0,1]^3 -> all samples can be used directly as texture coordinates
    let intersection_record = intersect_aabb(ray, AABB (float3(0.), float3(1.)));

    // early out if we missed the volume
    if !intersection_record.hit {
        return;
    }

    let start = ray_at(ray, max(0., intersection_record.t_min));
    let end   = ray_at(ray, intersection_record.t_max);

    let distance = distance(start * volume_scale, end * volume_scale);
    let num_steps = i32(distance / relative_step_size + 0.5);

    let max_distance = length(max(float3(1.) * volume_scale, float3(0.)));

    // early out if there is not a single sample within the volume
    if num_steps < 1 {
        return;
    }

    let step = (end - start) / f32(num_steps);

    let color = dvr(start, step, num_steps, ray);
    if color.a > 0.1 {
        let blended = blend(color, background_color);
        // todo: this shader should actually just render a volume block, not compose an image, so this should be done in a post-processing stage
        textureStore(result, pixel, float4(blended.rgb, 1.));
    }
}

// volume rendering stuff

fn max_steps_for_size(size: vec3<f32>) -> i32 {
    return i32(ceil((size.x * size.y * size.z) * sqrt(3.)));
}

fn dvr(start: float3, step: float3, num_steps: i32, ray: Ray) -> float4 {
    var color = float4(0.);

    var sample_location = start;
    for (var i: i32 = 0; i < num_steps; i += 1) {
        let value = sample_volume(sample_location);

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

fn central_differences(x: float3, step: float3) -> float3 {
    var central_differences = float3();
    for (var i = 0; i < 3; i += 1) {
        let h = step[i];
        var offset = float3();
        offset[i] = h;
        central_differences[i] = sample_volume(x - offset) - sample_volume(x + offset) / (2. * h);
    }
    return central_differences;
}

fn max_in_neighborhood(x: float3, step: float3) -> f32 {
    var max_value = 0.;
    for (var i = 0; i < 3; i += 1) {
        let h = step[i];
        var offset = float3();
        offset[i] = h;
        max_value = max(max_value, max(sample_volume(x - offset), sample_volume(x + offset)));
    }
    return max_value;
}

/// Computes the normal at point x from the central difference.
fn compute_volume_normal(x: float3, step: float3, view: float3) -> float3 {
    let normal = normalize(central_differences(x, step));
    // flip normal towards viewer if necessary
    let front_or_back = 2. * f32(dot(normal, view) > 0.) - 1.;
    return front_or_back * normal;
}

fn compute_lighting(value: f32, x: float3, step: float3, view_direction: float3) -> float4 {
    let view = normalize(view_direction);

    var light_direction = view;//float3(1.);

    let normal = compute_volume_normal(x, step, view);

    let ambient = 0.2;

    let diffuse = clamp(dot(normal, light_direction), 0., 1.);

    let halfway = normalize(light_direction + view);
    let shininess = 40.;
    let specular = pow(max(dot(halfway, normal), 0.), shininess);

    // Calculate final color by componing different components
    return float4(
        apply_colormap(value) * (ambient + diffuse) + specular,
        value
    );
}

fn apply_colormap(value: f32) -> float3{
    let u_clim = float2(0., 0.5);
    return uniforms.volume_color.rgb * (value - u_clim[0]) / (u_clim[1] - u_clim[0]);
}

// camera debugging

fn add_sphere(ray: Ray, pixel: int2, sphere: Sphere, color: float3) {
    let hit = intersect_sphere(ray, sphere);
    if hit.hit {
        let normal = normalize(ray_at(ray, hit.t_min) - sphere.center);
        let light_dir = normalize(float3(1.));
        let lighting = dot(normal, light_dir) * color;
        textureStore(result, pixel, float4(lighting, 1.));
    }
}

fn add_camera_debug_spheres(ray: Ray, pixel: int2, radius: f32) {
    let distance = radius * 4.;
    add_sphere(ray, pixel, Sphere(float3(), radius), float3(1.));
    add_sphere(ray, pixel, Sphere(float3(-1., -1., 0.) * distance, radius), float3(1., 0., 0.));
    add_sphere(ray, pixel, Sphere(float3(1., 1., 0.) * distance, radius), float3(0., 1., 0.));
    add_sphere(ray, pixel, Sphere(float3(-1., 1., 0.) * distance, radius), float3(0., 0., 1.));
    add_sphere(ray, pixel, Sphere(float3(1., -1., 0.) * distance, radius), float3(0.5, 0.5, 0.));
}
