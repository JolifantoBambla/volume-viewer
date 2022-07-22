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

// constant values
// todo: replace let with const
// no idea how infinity is written out in WGSL, so here are some constants
// note: no compile-time expressions without const -> needs to be a function...
fn positive_infinity() -> f32 { return  1. / 0.; }
fn negative_infinity() -> f32 { return -1. / 0.; }
// todo: maybe just remove this
let relative_step_size: f32 = 1.;
// todo: should be a uniform
let background_color = float4(float3(0.2), 1.);

// Colors for debugging
let RED = float4(1., 0., 0., 1.);
let GREEN = float4(0., 1., 0., 1.);
let BLUE = float4(0., 0., 1., 1.);
let WHITE = float4(1.);
let BLACK = float4(0., 0., 0., 1.);

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

struct Transform {
    // Maps a point in the object's space to a common world space.
    object_to_world: float4x4,

    // Maps a point in a common world space to the object's space.
    // It is the inverse of `object_to_world` (WGSL doesn't have a built-in function to compute the inverse of a matrix).
    world_to_object: float4x4,
}

type CameraType = u32;
let PERSPECTIVE = 0u;
let ORTHOGRAPHIC = 1u;

struct Camera {
    // Maps points from/to the camera's space to/from a common world space.
    transform: Transform,

    // Projects a point in the camera's object space to the camera's image plane.
    // This field is only needed for surface rendering.
    projection: float4x4,

    // The inverse of the `projection` matrix.
    // (WGSL doesn't have a built-in function to compute the inverse of a matrix).
    inverse_projection: float4x4,

    // The type of this camera:
    //  1:  Orthographic
    //  else: Perspective
    // Note that this field has a size of 16 bytes to avoid alignment issues.
    @size(16) camera_type: CameraType,
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

struct Timestamps {
    // Note: right now I'm thinking about going the Sarton route with one entry per brick in the multi-res volume / entry in cache
    // in that case, the cache entries don't have to be atomic since all threads write the same timestamp anyway
    cache: array<u32>,
}

struct PageTableMeta {
    max_lod: u32,
    brick_size: uint3,
}

// Bindings

// The bindings in group 0 should never change (except maybe the result image for double buffering?)
@group(0) @binding(0) var<uniform> page_table_meta: PageTableMeta;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var volume_sampler: sampler;
@group(0) @binding(3) var result_image: texture_storage_2d<rgba8unorm, write>;

// Each channel in the multiresolution, multichannel volume is represented by its own bind group of the following structure:
// 1) page_directory: holds page entries for accessing the brick cache
// 2) brick_cache: holds cached bricks
// 3) lru_cache: A Least-Recently-Used (LRU) cache containing one timestamp per brick in the brick cache.
// 4) request_buffer: A request buffer containing one timestamp per brick in the multi-resolution volume.
//    Each timestamp refers to the most recent time the brick with the corresponding index has been requested.
//    The default value 0 indicates that a brick has never been requested.

// Bind group 1 holds all data for one channel of a multiresolution, multichannel volume
@group(1) @binding(0) var page_directory: texture_3d<f32>;
@group(1) @binding(1) var brick_cache: texture_3d<f32>;
@group(1) @binding(2) var<storage> lru_cache: Timestamps;
@group(1) @binding(3) var<storage> request_buffer: Timestamps;


// Helper stuff

fn sample_volume(x: float3) -> f32 {
    return f32(textureSampleLevel(volume_data, volume_sampler, x, 0.).x);
}

/// An Axis Aligned Bounding Box (AABB)
struct AABB {
    @size(16) min: float3,
    @size(16) max: float3,
}

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

// todo: move this to a separate file that is included via a custom include mechanism

fn raster_to_screen(pixel: float2, resolution: float2) -> float2 {
    let aspect_ratio = resolution.x / resolution.y;
    var screen_min = float2(-aspect_ratio, -1.);
    var screen_max = float2( aspect_ratio,  1.);
    if aspect_ratio < 1. {
        var screen_min = float2(-1., -1. / aspect_ratio);
        var screen_max = float2( 1.,  1. / aspect_ratio);
    }
    return float2(
        pixel.x * ((screen_max.x - screen_min.x) / resolution.x) - screen_max.x,
        pixel.y * ((screen_min.y - screen_max.y) / resolution.y) + screen_max.y
    );
}

// Generates a ray in a common "world" space from a `Camera` instance.
fn generate_camera_ray(camera: Camera, pixel: float2, resolution: float2) -> Ray {
    let offset = 0.5;
    let camera_point = (camera.inverse_projection * float4(raster_to_screen(pixel + offset, resolution), 0., 1.)).xyz;

    var origin = float3();
    var direction = float3();

    if camera.camera_type == ORTHOGRAPHIC {
        origin = camera_point;
        direction = float3(0., 0., -1.);
    } else {
        direction = normalize(camera_point);
    }

    return transform_ray(
        Ray (origin, direction, positive_infinity()),
        camera.transform.object_to_world
    );
}

// Main stage

// supported builtins:
// - local_invocation_id    : vec3<u32>
// - local_invocation_index : u32
// - global_invocation_id   : vec3<u32>
// - workgroup_id           : vec3<u32>
// - num_workgroups         : vec3<u32>
@stage(compute)
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

// Ray methods

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
    return ray.origin + ray.direction * t;
}

// ray-box intersection

struct Intersection {
    hit: bool,
    t_min: f32,
    t_max: f32,
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



// utils
fn swap(a: ptr<function, f32, read_write>, b: ptr<function, f32, read_write>) {
    let helper = *a;
    *a = *b;
    *b = helper;
}

fn blend(a: float4, b: float4) -> float4 {
    return float4(
        a.rgb * a.a + b.rgb * b.a * (1. - a.a),
        a.a + b.a * (1. - a.a)
    );
}

// page table stuff
fn select_level_of_detail(f32 distance) -> u32 {
    // todo: select based on distance to camera
    return page_table_meta.max_lod;
}

fn aabb_contains(aabb: AABB, point: float3) -> bool {
    return all(point > aabb.min) && all(point < aabb.max);
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}

struct Brick {
    has_data: bool,
    origin: float3,

}



fn ray_guided_volume_rendering(volume_bounds: AABB) {
    var current_lod: u32;
    var current_step_size: float3;
    var current_step: float3;
    var current_depth_step_size: f32;
    var current_depth: f32;
    var current_color: float4;
    while (
        aabb_contains(volume_bounds, current_step) &&
        !is_saturated(current_color)
    ) {
        let lod = select_level_of_detail(current_depth);
        if lod != current_lod {
            current_lod = lod;
            // todo: increase step_size
            // todo: increase depth step size
        }
        let brick = maybe_fetch_brick(lod);

        if brick.has_data {
            ray_cast(ray, )
        }

        // we march through the volume in the volume's space, but need to keep track of the depth in the camera's space
        current_step  += current_step_size;
        current_depth += current_depth_step_size;
    }
}