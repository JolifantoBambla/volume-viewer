//

@include(aabb)
@include(camera)
@include(constant)
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
struct Uniforms {
    camera: Camera,
    volume_transform: Transform,
    @size(16) timestamp: u32,
}

// Bindings

// The bindings in group 0 should never change (except maybe the result image for double buffering?)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
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

/// Reports usage of a cached brick by storing the current frame's timestamp in the `brick_usage_buffer`.
///
/// # Arguments
/// * `brick_address`: the address of the cached brick. The actual brick is located at `brick_address` * `brick_size`
fn report_usage(brick_address: int3) {
    textureStore(brick_usage_buffer, brick_address, uint4(uniforms.timestamp));
}

/// Requests a brick missing in the `brick_cache` by storing the current frame's timestamp in the `request_buffer`.
///
/// # Arguments
/// * `page_address`:
fn request_brick(page_address: int3) {
    // todo: translate to actual location
    textureStore(request_buffer, page_address, uint4(uniforms.timestamp));
}

// todo: this could be a per-resolution constant
fn volume_to_padded(level: u32) -> float3 {
    let resolution = page_table_meta.resolutions[level];

    let volume_size = float3(resolution.volume_size);

    let extent = resolution.page_table_extent;
    let brick_size = resolution.brick_size;
    let padded_size = float3(brick_size * extent);

    return volume_size / padded_size;
}

fn compute_page_address(position: float3, level: u32) -> uint3 {
    let resolution = page_table_meta.resolutions[level];
    let offset = resolution.page_table_offset;
    let extent = resolution.page_table_extent;
    return min(
        offset + uint3(floor(float3(extent) * position)),
        offset + extent - uint3(1u)
    );
}

fn compute_cache_address(position: float3, level: u32, brick_address: uint3) -> uint3 {
    let resolution = page_table_meta.resolutions[level];
    let brick_size = resolution.brick_size;
    let volume_size = resolution.volume_size;
    return brick_address + max(uint3(ceil(position * float3(volume_size))), uint3(1u)) % brick_size;
}

fn normalize_cache_address(cache_address: uint3) -> float3 {
    let cache_size = float3(textureDimensions(brick_cache));
    return clamp(
        float3(cache_address) / cache_size,
        float3(),
        float3(1.)
    );
}

fn compute_offset(location: float3, level: u32) -> float3 {
    let resolution = page_table_meta.resolutions[level];
    let scale = float3(resolution.volume_size) / float3(resolution.brick_size);
    return floor(location * scale) / scale;
}

fn compute_brick_entry_and_exit(location: float3, level:u32, direction: float3, page: PageTableEntry) {
    let page_offset = compute_offset(location, level);
    let brick_offset = (float3(page.location) / float3(textureDimensions(brick_cache).xyz));
    let entry = location - page_offset + brick_offset;

}

fn transform_to_brick(location: float3, level: u32, page: PageTableEntry) -> float3 {
    // todo: check if that's correct
    return location - compute_offset(location, level) + (float3(page.location) / float3(textureDimensions(brick_cache).xyz));
}

fn get_brick_exit(location: float3, level: u32, page: PageTableEntry) -> float3 {
    // todo: check if that's correct
    // it's not -> assumes rays always go in same direction
    let resolution = page_table_meta.resolutions[level];
    let scale = float3(resolution.volume_size) / float3(resolution.brick_size);
    return ceil(location * scale) / scale + (float3(page.location) / float3(textureDimensions(brick_cache).xyz));
}

// todo: add mutliple virtualization levels
fn get_page(page_address: uint3) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, int3(page_address), 0));
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: uint3) {
    // terminate thread if it's outside the window bounds
    let window_size = uint2(textureDimensions(result));
    if (any(window_size < global_id.xy)) {
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
    if (!intersection_record.hit) {
        return;
    }

    let object_to_view = uniforms.camera.transform.world_to_object * uniforms.volume_transform.object_to_world;

    let start_os = clamp(ray_at(ray_os, max(0., intersection_record.t_min)), float3(), float3(1.));
    let end_os   = clamp(ray_at(ray_os, intersection_record.t_max), float3(), float3(1.));

    let timestamp = uniforms.timestamp;

    let lowest_lod = arrayLength(&page_table_meta.resolutions);
    let brick_size = page_table_meta.resolutions[0].brick_size;

    var color = float4();
    var current_lod = lowest_lod + 1u;
    var lod_volume_to_padded = 1.;

    // initialize line traversal (lod independent stuff)
    // if d[i] = 0, i = 0,1,2, brick_step[i] = 0, but this should not matter since we then only multiply 0 by 0 anyway
    let brick_step = sign(ray.direction);
    // updated on lod change
    var last_brick = uint3(1u);
    var t_delta = float3();

    // updated every step
    var current_position = start_os;
    var current_brick = uint3();
    var t_min = max(0., intersection_record.t_min);
    var t_max = float3(intersection_record.t_max);

    while (t_min < 0. || t_min > 1.) {
        let position = current_position;
        let distance_to_camera = abs((object_to_view * float4(position, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, lowest_lod);

        if (lod != current_lod) {
            current_lod = lod;
            lod_volume_to_padded = volume_to_padded(lod);

            current_brick = compute_page_address(position * lod_volume_to_padded, lod);

            let adjusted_end = stop_os * lod_volume_to_padded;
            last_brick = compute_page_address(adjusted_end, lod);

            let lod_meta = page_table_meta.resolutions[lod];
            let brick_size_spatial = 1. / float3(lod_meta.page_table_extent);
            t_delta = brick_size_spatial / ray.direction * brick_step;

            let previous_brick = current_brick - uint3(clamp(int3(brick_step), int3(-1), int3(0)) * -1);
            t_max = t_min + (float3(previous_brick) * brick_size_spatial - current_position) / ray.direction;
        }

        let page_address = compute_page_address(adjusted_position, lod);
        let page_color = float3(page_address) / float3(7., 7., 1.);

        let page = get_page(page_address);
        if (page.flag == UNMAPPED) {
            // color = RED;
            color = float4(page_color, 1.);
            request_brick(int3(page_address));
            break;
        } else if (page.flag == EMPTY) {
            color = BLUE;
        } else if (page.flag == MAPPED) {
            // todo: step through brick

            if (page.flag == MAPPED) {
                //color = GREEN;
                color = float4(page_color, 1.);
            }

            report_usage(int3(page.location / brick_size));

            let start = transform_to_brick(start_os, lod, page);
            let stop = get_brick_exit(start_os, lod, page);
            let brick_distance = distance(start * float3(page_table_meta.resolutions[lod].brick_size), stop * float3(page_table_meta.resolutions[lod].brick_size));
            let num_steps = i32(brick_distance + 0.5);
            if (num_steps < 1) {
                color = WHITE;
                break;
            }

            let step = (stop - start) / f32(num_steps);
            //color = ray_cast(color, start, step, num_steps);
            color = float4(float3(sample_volume(adjusted_position)), 1.);

            /*
            // todo: this is not true - needs to be translated
            let start = start_os;
            // todo: this is not true - needs to be scaled (does it though)
            let step = ray.direction;
            // todo: this is not true - needs to be determined based on brick boundary
            let num_steps = 5;
            color = ray_cast(color, start, step, num_steps);
            */

            if (is_saturated(color)) {
                break;
            }
        } else {
            let s = sample_volume(float3());
            if (s == 0.) {
                color = RED;
            }
        }


        // todo: float operations should be = tmaxOriginal + n * tDelta
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
            current_brick += brick_step.x;
            t_min = t_max.x;
            t_max.x += t_delta.x;
        } else if (t_max.y < t_max.z) {
            current_brick += brick_step.y;
            t_min = t_max.y;
            t_max.y += t_delta.y;
        } else {
            current_brick += brick_step.z;
            t_min = t_max.z;
            t_max.z += t_delta.z;
        }
        current_position = clamp(ray_at(ray_os, t_min), float3(), float3(1.));
    }

    // todo: this should be in a loop:
    for (var i = 0; i < 1; i += 1) {
        let position = current_position;
        let distance_to_camera = abs((object_to_view * float4(position, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, lowest_lod);

        // todo: scale needs to take under-full pages into account!
        let adjusted_position = position * volume_to_padded(lod);

        // todo: this all works in floats, because it's faster but doing it in ints would be safer
        if (lod != current_lod) {
            current_lod = lod;
            lod_volume_to_padded = volume_to_padded(lod);

            current_brick = compute_page_address(adjusted_position, lod);

            let adjusted_end = stop_os * lod_volume_to_padded;
            last_brick = compute_page_address(adjusted_end, lod);

            let lod_meta = page_table_meta.resolutions[lod];
            let brick_size_spatial = 1. / float3(lod_meta.page_table_extent);
            t_delta = brick_size_spatial / ray.direction * brick_step;

            let previous_brick = current_brick - uint3(clamp(int3(brick_step), int3(-1), int3(0)) * -1);
            t_max = t_min + (float3(previous_brick) * brick_size_spatial - current_position) / ray.direction;
        }

        let page_address = compute_page_address(adjusted_position, lod);
        let page_color = float3(page_address) / float3(7., 7., 1.);

        let page = get_page(page_address);
        if (page.flag == UNMAPPED) {
            // color = RED;
            color = float4(page_color, 1.);
            request_brick(int3(page_address));
        } else if (page.flag == EMPTY) {
            color = BLUE;
            //debug(pixel, float4(normalize(float3(page_address)), 1.));
        } else if (page.flag == MAPPED) {
            // todo: step through brick

            if (page.flag == MAPPED) {
                //color = GREEN;
                color = float4(page_color, 1.);
            }

            report_usage(int3(page.location / brick_size));

            let start = transform_to_brick(start_os, lod, page);
            let stop = get_brick_exit(start_os, lod, page);
            let brick_distance = distance(start * float3(page_table_meta.resolutions[lod].brick_size), stop * float3(page_table_meta.resolutions[lod].brick_size));
            let num_steps = i32(brick_distance + 0.5);
            if (num_steps < 1) {
                color = WHITE;
                break;
            }

            let step = (stop - start) / f32(num_steps);
            //color = ray_cast(color, start, step, num_steps);
            color = float4(float3(sample_volume(adjusted_position)), 1.);

            /*
            // todo: this is not true - needs to be translated
            let start = start_os;
            // todo: this is not true - needs to be scaled (does it though)
            let step = ray.direction;
            // todo: this is not true - needs to be determined based on brick boundary
            let num_steps = 5;
            color = ray_cast(color, start, step, num_steps);
            */

            if (is_saturated(color)) {
                break;
            }
        } else {
            let s = sample_volume(float3());
            if (s == 0.) {
                color = RED;
            }
        }
    }
    debug(pixel, color);

    /*
    MAYBE GET BRICK:
        GET BRICK AND REPORT USAGE OR REQUEST

    ENTER LOOP UNTIL EXIT OR SATURATED
        DETERMINE LOD
        MAYBE GET BRICK
        IF BRICK MAPPED
            STEP THROUGH BRICK
    STORE RESULT
    */
}

fn ray_cast(in_color: float4, start: float3, step: float3, num_steps: i32) -> float4 {
    var color = in_color;

    var sample_location = start;
    for (var i = 0; i < num_steps; i += 1) {
        let value = sample_volume(sample_location);

        // todo: make minimum threshold configurable
        if (value > 0.1) {
            // todo: compute lighting
            var lighting = BLUE;
            lighting.a = value;
            color += lighting;
        }

        if (is_saturated(color)) {
            break;
        }

        sample_location += step;
    }

    return color;
}



// page table stuff
fn select_level_of_detail(distance: f32, lowest_lod: u32) -> u32 {
    // todo: select based on distance to camera or screen size?
    return 0u; //page_table_meta.max_lod;
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}
