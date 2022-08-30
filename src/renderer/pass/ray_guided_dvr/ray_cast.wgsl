//

@include(aabb)
@include(camera)
@include(constant)
@include(page_table)
@include(ray)
@include(transform)
@include(type_alias)
@include(voxel_line_f32)

// includes that require the shader to define certain functions
@include(volume_util)

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

fn clone_page_table_meta(level: u32) -> PageTableMeta {
    let page_table_meta = page_table_meta.resolutions[level];
    return PageTableMeta(
        page_table_meta.brick_size,
        page_table_meta.page_table_offset,
        page_table_meta.page_table_extent,
        page_table_meta.volume_size
    );
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

/*
fn compute_page_address(position: float3, level: u32) -> uint3 {
    let resolution = page_table_meta.resolutions[level];
    let offset = resolution.page_table_offset;
    let extent = resolution.page_table_extent;
    return min(
        offset + uint3(floor(float3(extent) * position)),
        offset + extent - uint3(1u)
    );
}
*/

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

fn compute_offset(position: float3, level: u32) -> float3 {
    let resolution = page_table_meta.resolutions[level];
    let scale = float3(resolution.volume_size) / float3(resolution.brick_size);
    return floor(position * scale) / scale;
}

fn transform_to_brick(position: float3, level: u32, page: PageTableEntry) -> float3 {
    // todo: check if that's correct
    return position - compute_offset(position, level) + (float3(page.location) / float3(textureDimensions(brick_cache).xyz));
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

    let t_min = max(0., intersection_record.t_min);
    let t_max = intersection_record.t_max;

    let timestamp = uniforms.timestamp;

    let lowest_lod = arrayLength(&page_table_meta.resolutions);
    let brick_size = page_table_meta.resolutions[0].brick_size;

    var color = float4();
    var requested_brick = false;
    var last_lod = 0u;
    var page_table = clone_page_table_meta(0u);

    for (
        var voxel_line = create_voxel_line(ray_os, t_min, t_max, &page_table);
        in_grid(&voxel_line);
        advance(&voxel_line, ray_os)
    ) {
        let position = voxel_line.state.entry;
        let distance_to_camera = abs((object_to_view * float4(position, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, lowest_lod);
        if (lod != last_lod) {
            last_lod = lod;
            page_table = clone_page_table_meta(lod);
            // todo: this could lead to problems if the line creation assumes that t_min is either 0 (in the grid) or x (at the grid boundary) while it might actually actually be y > 0 (somewhere in the grid)
            voxel_line = create_voxel_line(ray_os, voxel_line.state.t_min, t_max, &page_table);
        }

        let page_address = uint3(voxel_line.state.brick);
        let page_color = float3(page_address) / float3(7., 7., 1.);

        let page = get_page(page_address);
        if (page.flag == UNMAPPED) {
            // todo: remove this (debug)
            color = float4(page_color, 1.);

            if (!requested_brick) {
                // todo: maybe request lower res as well?
                request_brick(int3(page_address));
                requested_brick = true;
            }

            /*
            if (any(voxel_line.state.t_max < float3(0.0))) {
                color = BLUE;
                break;
            }

            if (any(voxel_line.state.t_max < float3(voxel_line.state.t_min))) {
                color = WHITE;
                break;
            }

            let brick_step = int3(sign(voxel_line.state.exit - voxel_line.state.entry));
            if (all(brick_step == voxel_line.brick_step)) {
                color = GREEN;
                break;
            } else {
                color = RED;
                break;
            }
            */
        } else if (page.flag == EMPTY) {
            // todo: remove this (debug)
            color = BLUE;
        } else if (page.flag == MAPPED) {
            if (page.flag == MAPPED) {
                //color = float4(page_color, 1.);
            }

            report_usage(int3(page.location / brick_size));

            let start = normalize_cache_address(compute_cache_address(voxel_line.state.entry, lod, page.location));
            let stop = normalize_cache_address(compute_cache_address(voxel_line.state.exit, lod, page.location));


            let brick_step = int3(sign(voxel_line.state.exit - voxel_line.state.entry));
            var correct = true;
            for (var i = 0; i < 3; i += 1) {
                if (brick_step[i] > 0) {
                    if (voxel_line.brick_step[i] <= 0) {
                        correct = false;
                    }
                } else if (brick_step[i] == 0) {
                    if (voxel_line.brick_step[i] != 0) {
                        correct = false;
                    }
                } else {
                    if (voxel_line.brick_step[i] >= 0) {
                        correct = false;
                    }
                }
            }
            if (!correct) {
                color = RED;
                break;
            }
            if (all(brick_step == voxel_line.brick_step)) {
                //color = GREEN;
                //break;
            }


            let brick_distance = distance(
                start * float3(page_table_meta.resolutions[lod].brick_size),
                stop * float3(page_table_meta.resolutions[lod].brick_size)
            );
            let num_steps = i32(brick_distance + 0.5);


            if (num_steps < 1) {
                //color = RED;
                break;
            }


            // todo: step through brick
            let step = (stop - start) / f32(num_steps);
            color = ray_cast(color, start, step, num_steps);
            break;
            //color = float4(float3(sample_volume(start)), 1.);

            if (is_saturated(color)) {
                break;
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
    let view_direction = normalize(step);
    var color = in_color;

    var sample_location = start;
    for (var i = 0; i < num_steps; i += 1) {
        let value = sample_volume(sample_location);

        // todo: make minimum threshold configurable
        if (value > 0.1) {
            // todo: compute lighting
            var lighting = compute_lighting(value, sample_location, step, view_direction);
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
    return BLUE.rgb * (value - u_clim[0]) / (u_clim[1] - u_clim[0]);
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
