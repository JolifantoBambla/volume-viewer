//

@include(aabb)
@include(camera)
@include(page_table)
@include(ray)
@include(transform)
@include(type_alias)

// constant values
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

// todo: come up with a better name...
// todo: clean up those uniforms - find out what i really need and throw away the rest
struct Transforms {
    camera: Camera,
    volume_transform: Transform,
}

// Bindings

// The bindings in group 0 should never change (except maybe the result image for double buffering?)
@group(0) @binding(0) var<uniform> uniforms: Transforms;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var result: texture_storage_2d<rgba8unorm, write>;

// Each channel in the multiresolution, multichannel volume is represented by its own bind group of the following structure:
// 1) page_directory: holds page entries for accessing the brick cache
// 2) brick_cache: holds cached bricks
// !out of date:
// 3) lru_cache: A Least-Recently-Used (LRU) cache containing one timestamp per brick in the brick cache.
// 4) request_buffer: A request buffer containing one timestamp per brick in the multi-resolution volume.
//    Each timestamp refers to the most recent time the brick with the corresponding index has been requested.
//    The default value 0 indicates that a brick has never been requested.

// Bind group 1 holds all data for one channel of a multiresolution, multichannel volume
@group(1) @binding(0) var<storage> page_table_meta: PageDirectoryMeta;
@group(1) @binding(1) var page_directory: texture_3d<u32>;
@group(1) @binding(2) var brick_cache: texture_3d<f32>;
@group(1) @binding(3) var brick_usage_buffer: texture_storage_3d<r32uint, write>;
@group(1) @binding(4) var request_buffer: texture_storage_3d<r32uint, write>;

// Helper stuff

fn sample_volume(x: float3) -> f32 {
    return f32(textureSampleLevel(brick_cache, volume_sampler, x, 0.).x);
}

@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    // terminate thread if it's outside the window bounds
    let window_size = uint2(textureDimensions(result));
    if any(window_size < global_id.xy) {
        return;
    }

    let pixel = int2(global_id.xy);
    let resolution = float2(window_size);

    // initialize result
    textureStore(result, pixel, float4());

    // generate a ray and transform it to the volume's space (i.e. where the volume is a unit cube with x in [0,1]^3)
    let ray_ws = generate_camera_ray(uniforms.camera, float2(pixel), resolution);
    let ray_os = transform_ray(ray_ws, uniforms.volume_transform.world_to_object);

    // terminate thread if the ray isn't within the unit cube
    let volume_bounds_os = AABB(float3(0.), float3(1.));
    let intersection_record = intersect_aabb(ray_os, volume_bounds_os);
    if !intersection_record.hit {
        return;
    }

    let object_to_view = uniforms.camera.transform.world_to_object * uniforms.volume_transform.object_to_world;

    let start_os = ray_at(ray_os, max(0., intersection_record.t_min));
    let end_os   = ray_at(ray_os, intersection_record.t_max);

    let s = sample_volume(float3());
    if s == 0. && page_table_meta.resolutions[0].brick_size.x == 0u {
        let page = u32(textureLoad(page_directory, int3(), 0).x);
        textureStore(brick_usage_buffer, int3(), uint4(page));
        textureStore(request_buffer, int3(), uint4(page));
    }

    green(pixel);
/*
    MAYBE GET BRICK:
        GET BRICK AND REPORT USAGE OR REQUEST

    ENTER LOOP UNTIL EXIT OR SATURATED
        DETERMINE LOD
        MAYBE GET BRICK
        IF BRICK MAPPED
            STEP THROUGH BRICK
    STORE RESULT

    // initialize to invalid value
    var current_lod = arrayLength(&page_table_meta.resolutions) + 1u;

    var current_position_os = start_os;
    var current_depth = length(object_to_view * current_position_os);

    var current_voxel = pt_canonical_to_voxel(page_table_meta, current_step_os, current_lod);
    let end_voxel = pt_canonical_to_voxel(page_table_meta, end_os, current_lod);

    var current_color = float4();
    var current_line = Line();
    while (
        !is_saturated(current_color) &&
        aabb_contains(volume_bounds_os, current_position_os)
    ) {
        let lod = select_level_of_detail(current_depth);
        if lod != current_lod {
            current_line = create_bresenham3d(
                start: int3,
                end: int3,
            );
            current_lod = lod;
            // todo: increase step_size
            // todo: increase depth step size
        }

        current_depth += current_depth_step;
        current_step +=
    }
    */
}

// page table stuff
fn select_level_of_detail(distance: f32) -> u32 {
    // todo: select based on distance to camera
    return 0u; //page_table_meta.max_lod;
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}
