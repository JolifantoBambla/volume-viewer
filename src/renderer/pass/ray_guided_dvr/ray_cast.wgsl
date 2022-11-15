//

@include(aabb)
@include(camera)
@include(constant)
@include(page_table)
@include(ray)
@include(timestamp)
@include(transform)
@include(type_alias)
@include(grid_traversal)

// includes that require the shader to define certain functions
@include(volume_util)

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

// RENDER MODES
const GRID_TRAVERSAL = 0u;
const DIRECT = 1u;

struct GlobalSettings {
    render_mode: u32,
    step_scale: f32,
    max_steps: u32,
    num_visible_channels: u32,
    background_color: float4,
}

// todo: come up with a better name...
// todo: clean up those uniforms - find out what i really need and throw away the rest
struct Uniforms {
    camera: Camera,
    volume_transform: Transform,
    //todo: use Timestamp struct
    @size(16) timestamp: u32,
    settings: GlobalSettings,
}

struct ChannelSettings {
    color: float4,
    channel_index: u32,
    max_lod: u32,
    min_lod: u32,
    threshold_lower: f32,
    threshold_upper: f32,
    visible: u32,
    page_table_index: u32,
    padding2: u32,
}

struct ChannelSettingsList {
    channels: array<ChannelSettings>,
}

// Bindings

// The bindings in group 0 should never change (except maybe the result image for double buffering?)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var result: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage> channel_settings_list: ChannelSettingsList;

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

fn load_voxel(x: uint3) -> f32 {
    return f32(textureLoad(brick_cache, int3(x), 0).x);
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

fn clone_channel_settings(channel_index: u32) -> ChannelSettings {
    let channel_settings = channel_settings_list.channels[channel_index];
    return ChannelSettings(
        channel_settings.color,
        channel_settings.channel_index,
        channel_settings.max_lod,
        channel_settings.min_lod,
        channel_settings.threshold_lower,
        channel_settings.threshold_upper,
        channel_settings.visible,
        channel_settings.page_table_index, // page_table_index,
        0u  // padding2,
     );
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

    let num_channels = uniforms.settings.num_visible_channels;// arrayLength(&channel_settings_list.channels);
    let num_resolutions = arrayLength(&page_table_meta.resolutions) / num_channels;

    // todo: look for visible channels as soon as multiple channels are supported
    let cs = clone_channel_settings(0);

    let lowest_res = min(cs.min_lod, num_resolutions);
    let highest_res = min(cs.max_lod, lowest_res);
    
    let brick_size = page_table_meta.resolutions[0].brick_size;

    // Set up state tracking
    var color = float4();
    var requested_brick = false;
    var last_lod = 0u;
    var steps_taken = 0u;

    if (uniforms.settings.render_mode == GRID_TRAVERSAL) {
        // todo: use active_page_tables instead
        var page_table = clone_page_table_meta(0u);

        for (
            var grid_traversal = create_grid_traversal(ray_os, t_min, t_max, page_table);
            in_grid(grid_traversal);
            next_voxel(&grid_traversal)
        ) {
            let brick_endpoints = compute_voxel_endpoints(grid_traversal);

            let position = brick_endpoints.entry;
            let distance_to_camera = abs((object_to_view * float4(position, 1.)).z);
            let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res);
            if (lod != last_lod) {
                last_lod = lod;
                page_table = clone_page_table_meta(lod);
                // todo: t_entry is w.r.t. grid_traversal.ray not ray_os!
                grid_traversal = create_grid_traversal(ray_os, grid_traversal.state.t_entry, t_max, page_table);
            }

            let page_address = uint3(grid_traversal.state.voxel);

            // todo: remove (debug)
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
            } else if (page.flag == MAPPED) {
                report_usage(int3(page.location / brick_size));

                let padded_size = float3(page_table.brick_size * page_table.page_table_extent);
                let entry = max(uint3(ceil(brick_endpoints.entry * float3(padded_size))), uint3(1u)) % brick_size;
                let exit  = max(uint3(ceil(brick_endpoints.exit  * float3(padded_size))), uint3(1u)) % brick_size;


                let start = normalize_cache_address(page.location + entry);
                let stop  = normalize_cache_address(page.location + exit);

                // todo: fix num_steps, find out why boundaries are visible
                let brick_distance = distance(
                    start * float3(page_table.brick_size),
                    stop  * float3(page_table.brick_size)
                );
                let num_steps = i32(brick_distance + 0.5);
                if (num_steps < 1) {
                    //color += RED;
                    //break;
                    continue;
                }

                let step = (stop - start) / f32(num_steps);
                color = ray_cast(color, start, step, num_steps, cs);

                if (is_saturated(color)) {
                    break;
                }

                steps_taken += 1;
            }
            if (steps_taken >= uniforms.settings.max_steps) {
                break;
            }
        }
    } else {
        let offset = wang_hash(pixel.x + 640 * pixel.y);
        let dt_scale = uniforms.settings.step_scale;

        var dt_vec = 1. / (float3(page_table_meta.resolutions[last_lod].volume_size) * abs(ray_os.direction));
        var dt = dt_scale * min_component(dt_vec);
        var p = ray_at(ray_os, t_min + offset * dt);

        for (var t = t_min; t < t_max; t += dt) {
            let distance_to_camera = abs((object_to_view * float4(p, 1.)).z);
            let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res);
            let lod_changed = lod != last_lod;
            if (lod_changed) {
                last_lod = lod;
                dt_vec = 1. / (float3(page_table_meta.resolutions[lod * num_channels].volume_size) * abs(ray_os.direction));
                dt = dt_scale * min_component(dt_vec);
            }

            for (var channel = 0u; channel < num_channels; channel += 1u) {
                let page_table = clone_page_table_meta(channel + lod * num_channels);

                // todo: think about this more carefully - does that change the ray or not?
                let position_corrected = p * compute_volume_to_padded(page_table);
                let page_address = compute_page_address(page_table, position_corrected);
                let page = get_page(page_address);

                // todo: remove (debug)
                let page_color = float3(page_address) / float3(7., 7., 1.);

                if (page.flag == UNMAPPED) {
                    // todo: remove this (debug)
                    color = float4(page_color, 1.);

                    if (!requested_brick) {
                        // todo: maybe request lower res as well?
                        request_brick(int3(page_address));
                        requested_brick = true;
                    }
                } else if (page.flag == MAPPED) {
                    report_usage(int3(page.location / brick_size));

                    let sample_location = normalize_cache_address(
                        compute_cache_address(page_table, p, page)
                    );
                    let value = sample_volume(sample_location);

                    // todo: make minimum threshold configurable
                    if (value >= channel_settings_list.channels[channel].threshold_lower && value <= channel_settings_list.channels[channel].threshold_upper) {
                        let trans_sample = channel_settings_list.channels[channel].color;
                        var val_color = float4(trans_sample.rgb, value * trans_sample.a);
                        val_color.a = 1.0 - pow(1.0 - val_color.a, dt_scale);
                        color += float4((1.0 - color.a) * val_color.a * val_color.rgb, 0.);
                        color.a += (1.0 - color.a) * val_color.a;
                    }

                    if (is_saturated(color)) {
                        break;
                    }
                }
            }

            steps_taken += 1;
            if (steps_taken >= uniforms.settings.max_steps) {
                break;
            }

            p += ray_os.direction * dt;
        }
    }

    debug(pixel, color);
}

fn ray_cast(in_color: float4, start: float3, step: float3, num_steps: i32, channel_settings: ChannelSettings) -> float4 {
    let view_direction = normalize(step);
    var color = in_color;

    var sample_location = start;
    for (var i = 0; i < num_steps; i += 1) {
        let value = sample_volume(sample_location);

        if (value >= channel_settings.threshold_lower && value <= channel_settings.threshold_upper) {
            var lighting = compute_lighting(value, sample_location, step, view_direction, channel_settings);
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

fn compute_lighting(value: f32, x: float3, step: float3, view: float3, channel_settings: ChannelSettings) -> float4 {
    var light_direction = view;//float3(1.);

    let normal = compute_volume_normal(x, step, view);

    let ambient = 0.2;

    let diffuse = clamp(dot(normal, light_direction), 0., 1.);

    let halfway = normalize(light_direction + view);
    let shininess = 16.;
    let specular = pow(max(dot(halfway, normal), 0.), shininess);

    return float4(
        apply_colormap(value, channel_settings) * (ambient + diffuse) + specular,
        value
    );
}

fn apply_colormap(value: f32, channel_settings: ChannelSettings) -> float3{
    let u_clim = float2(0., 0.5);
    return channel_settings.color.rgb * (value - u_clim[0]) / (u_clim[1] - u_clim[0]);
}

// page table stuff
fn select_level_of_detail(distance: f32, highest_res: u32, lowest_res: u32) -> u32 {
    // todo: select based on distance to camera or screen size?
    // let lod ...
    // return clamp(lod, highest_res, lowest_res);
    return 0u; //page_table_meta.max_lod;
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}
