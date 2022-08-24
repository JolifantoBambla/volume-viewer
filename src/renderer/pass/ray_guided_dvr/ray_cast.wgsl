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

fn get_page_address_old(location: float3, level: u32) -> uint3 {
    let resolution = page_table_meta.resolutions[level];
    let offset = resolution.page_table_offset;
    let max_local_subscript = resolution.page_table_extent - uint3(1u);
    return min(
        offset + max_local_subscript,
        max(
            offset + uint3(ceil(float3(max_local_subscript) * location)),
            offset
        )
    );
}

fn get_page_address(location: float3, level: u32) -> uint3 {
    let resolution = page_table_meta.resolutions[level];
    let offset = resolution.page_table_offset;
    let extent = resolution.page_table_extent;
    let max_local_subscript = extent - uint3(1u);
    return min(
        offset + max_local_subscript,
        offset + uint3(floor(float3(extent) * location))
    );
}

fn compute_offset(location: float3, level: u32) -> float3 {
    let resolution = page_table_meta.resolutions[level];
    let scale = float3(resolution.volume_size) / float3(resolution.brick_size);
    return floor(location * scale) / scale;
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

fn get_page(page_address: uint3) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, int3(page_address), 0));
}

@compute
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

    let start_os = clamp(ray_at(ray_os, max(0., intersection_record.t_min)), float3(), float3(1.));
    let end_os   = clamp(ray_at(ray_os, intersection_record.t_max), float3(), float3(1.));

    let timestamp = uniforms.timestamp;

    let lowest_lod = arrayLength(&page_table_meta.resolutions);
    let brick_size = page_table_meta.resolutions[0].brick_size;

    var color = float4();

    // translate from lod loc to cache loc:
    //    ((x - brick_offset_lod) * ( (brick_size_lod^2) / (vol_size_lod * brick_size_cache)) + brick_offset_cache)
    // constant per lod: ( (brick_size_lod^2) / (vol_size_lod * brick_size_cache))
    // brick_size_lod ... the brick size at that resolution level
    // brick_size_cache ... the brick size in cache
    // brick_offset_lod ... the offset of the brick x is in at that resolution level
    // brick_offset_cache ... the offset of the brick in cache
    // vol_size_lod ... the volume size at that lod
    // e.g.
    // brick_size_cache = 128
    // vol_size = [1024, 512, 256, 128]
    // brick_size = vol_size / brick_size_cache
    //
    /*
>>> brick_size_cache = 128
>>> vol_size = [1024, 512, 256, 128]
>>> brick_size = [i / brick_size_cache for i in vol_size]
>>> brick_size
[8.0, 4.0, 2.0, 1.0]
>>> c = [pow(brick_size[i], 2) / (vol_size[i] * brick_size_cache) for i in [0,1,2,3]]
>>> c
[0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05]
>>> floor(0.263 * 1024 / brick_size_cache)
2
>>> x = 0.263
>>> def to_lod_offset(x, lod):
...     return floor(x * brick_size[lod]) / brick_size[lod]
...
>>> def to_cache(x, lod):
...     return (x - to_lod_offset(x, lod)) * c[lod] + to_lod_offset(x, 0)


>>> (x-to_lod_offset(x, lod)) / (brick_size[lod] * brick_size_cache / vol_size[lod]) + to_lod_offset(x, 0)
okay, I think it's just:
    x - to_lod_offset(x, lod) + page_table_entry.offset
    to_lod_offset = floor(x * (vol_size[lod]/brick_size_cache)) / (vol_size[lod]/brick_size_cache)
    */

    // todo: scale needs to take under-full pages into account!

    // todo: this should be in a loop:
    for (var i = 0; i < 1; i += 1) {
        let location = start_os;
        let distance_to_camera = abs((object_to_view * float4(location, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, lowest_lod);

        let page_address = get_page_address(location, lod);

        let page_color = float3(page_address) / float3(7., 7., 1.);

        let page = get_page(page_address);
        if page.flag == UNMAPPED {
            // color = RED;
            color = float4(normalize(float3(page_address)), 1.);
            request_brick(int3(page_address));
        } else if page.flag == EMPTY {
            color = BLUE;
            //debug(pixel, float4(normalize(float3(page_address)), 1.));
        } else if page.flag == MAPPED {
            // todo: step through brick

            if page.flag == MAPPED {
                //color = GREEN;
                //color = float4(page_color, 1.);
            }

            report_usage(int3(page.location / brick_size));

            let start = transform_to_brick(start_os, lod, page);
            let stop = get_brick_exit(start_os, lod, page);
            let brick_distance = distance(start * float3(page_table_meta.resolutions[lod].brick_size), stop * float3(page_table_meta.resolutions[lod].brick_size));
            let num_steps = i32(brick_distance + 0.5);
            if num_steps < 1 {
                color = WHITE;
                break;
            }

            let step = (stop - start) / f32(num_steps);
            //color = ray_cast(color, start, step, num_steps);
            color = float4(float3(sample_volume(start_os)), 1.);

            /*
            // todo: this is not true - needs to be translated
            let start = start_os;
            // todo: this is not true - needs to be scaled (does it though)
            let step = ray.direction;
            // todo: this is not true - needs to be determined based on brick boundary
            let num_steps = 5;
            color = ray_cast(color, start, step, num_steps);
            */

            if is_saturated(color) {
                break;
            }
        } else {
            let s = sample_volume(float3());
            if s == 0. {
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
        if value > 0.1 {
            // todo: compute lighting
            var lighting = BLUE;
            lighting.a = value;
            color += lighting;
        }

        if is_saturated(color) {
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
