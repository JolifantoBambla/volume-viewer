//

@include(aabb)
@include(camera)
@include(channel_settings)
@include(constant)
@include(global_settings)
@include(grid_leap)
@include(lighting)
@include(output_modes)
@include(page_table)
@include(page_directory_meta_util)
@include(ray)
@include(timestamp)
@include(transform)
@include(type_alias)

// includes that require the shader to define certain functions
@include(volume_util)

// Note: volume_accelerator needs to map to some file!
@include(volume_accelerator)

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

// Bindings

// The bindings in group 0 should never change (except maybe the result image for double buffering?)
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var result: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage> channel_settings: array<ChannelSettings>;

// Each channel in the multiresolution, multichannel volume is represented by its own bind group of the following structure:
// 1) page_directory: holds page entries for accessing the brick cache
// 2) brick_cache: holds cached bricks
// !out of date:
// 3) lru_cache: A Least-Recently-Used (LRU) cache containing one timestamp per brick in the brick cache.
// 4) request_buffer: A request buffer containing one timestamp per brick in the multi-resolution volume.
//    Each timestamp refers to the most recent time the brick with the corresponding index has been requested.
//    The default value 0 indicates that a brick has never been requested.

// Bind group 1 holds all data for one channel of a multiresolution, multichannel volume
@group(1) @binding(0) var<uniform> page_directory_meta: PageDirectoryMeta;
@group(1) @binding(1) var<storage> page_table_meta: PageTableMetas;
@group(1) @binding(2) var page_directory: texture_3d<u32>;
@group(1) @binding(3) var brick_cache: texture_3d<f32>;
@group(1) @binding(4) var brick_usage_buffer: texture_storage_3d<r32uint, write>;
@group(1) @binding(5) var request_buffer: texture_storage_3d<r32uint, write>;

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
    textureStore(request_buffer, page_address, uint4(uniforms.timestamp));
}

fn clone_channel_settings(channel_index: u32) -> ChannelSettings {
    let cs = channel_settings[channel_index];
    return ChannelSettings(
        cs.color,
        cs.channel_index,
        cs.max_lod,
        cs.min_lod,
        cs.threshold_lower,
        cs.threshold_upper,
        cs.visible,
        cs.page_table_index,
        cs.lod_factor
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

// todo: add mutliple virtualization levels
fn get_page(page_address: uint3) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, int3(page_address), 0));
}

fn compute_page_table_index(channel: u32, lod: u32) -> u32 {
    return channel + lod * page_directory_meta.max_channels;
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

    let center_to_pixel = length(vec2<f32>(pixel) - (resolution / 2.0));
    let center_to_corner = length(resolution / 2.0);
    let pixel_radius = center_to_pixel / center_to_corner;
    let request_bricks = pixel_radius <= uniforms.settings.brick_request_radius;

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

    let num_visible_channels = uniforms.settings.num_visible_channels;
    let num_resolutions = page_directory_meta.max_resolutions;

    let cs = clone_channel_settings(0);

    let inv_high_res_size = 1. / float3(page_table_meta.metas[0].volume_size);
    let brick_size = page_table_meta.metas[0].brick_size;

    let lowest_res = min(cs.min_lod, num_resolutions - 1);
    let highest_res = min(cs.max_lod, lowest_res);
    let lod_factor = cs.lod_factor * max_component(inv_high_res_size);

    // Set up state tracking
    var color = float4();
    var requested_brick = !request_bricks;//false;
    var last_lod = 0u;
    var steps_taken = 0u;
    var bricks_accessed = 0u;
    var nodes_accessed = 0u;

    let offset = hash_u32_to_f32(pcg_hash(u32(pixel.x + pixel.x * pixel.y)));
    let dt_scale = uniforms.settings.step_scale;

    var dt_vec = 1. / ((float3(page_table_meta.metas[last_lod].volume_size) * uniforms.settings.voxel_spacing) * abs(ray_os.direction));
    var dt = dt_scale * min_component(dt_vec);
    var p = ray_at(ray_os, t_min + offset * dt);

    let first_channel_index = channel_settings[0].page_table_index;

    for (var t = t_min; t < t_max; t += dt) {
        let distance_to_camera = abs((object_to_view * float4(p, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res, lod_factor);
        let lod_changed = lod != last_lod;
        if (lod_changed) {
            last_lod = lod;
            dt_vec = 1. / ((float3(page_table_meta.metas[compute_page_table_index(first_channel_index, lod)].volume_size) * uniforms.settings.voxel_spacing) * abs(ray_os.direction));
            dt = dt_scale * min_component(dt_vec);
        }

        var empty_channels = 0u;
        var empty_lod = lod;

        for (var channel = 0u; channel < num_visible_channels; channel += 1u) {
            nodes_accessed += 1u;
            let channel_lod = va_compute_lod(distance_to_camera, channel, channel_settings[channel].lod_factor * max_component(inv_high_res_size));
            empty_lod = max(empty_lod, channel_lod);

            let node = va_get_node(p, channel_lod, channel, !requested_brick);
            requested_brick = requested_brick || node.requested_brick;
            if (node.is_mapped) {
                bricks_accessed += 1u;

                // todo: this currently means that the brick is empty, but the whole node thing should get refactored
                if (node.has_average) {
                    empty_channels += 1;
                    continue;
                }

                let value = sample_volume(node.sample_address);
                compute_lighting(value, channel, dt_scale, &color);
                if (is_saturated(color)) {
                    break;
                }
            }
        }

        steps_taken += 1;
        if (steps_taken >= uniforms.settings.max_steps) {
            break;
        }
        if (is_saturated(color)) {
            break;
        }

        if (empty_channels == num_visible_channels) {
            let dt_jump = compute_brick_jump_distance(p, empty_lod, dt, t_max, ray_os);
            p += ray_os.direction * dt_jump;
            t += dt_jump;
        }

        p += ray_os.direction * dt;
    }

    if (uniforms.settings.output_mode == DVR) {
        debug(pixel, color);
    } else if (uniforms.settings.output_mode == BRICKS_ACCESSED) {
        debug(pixel, vec4<f32>(vec3(f32(bricks_accessed) / f32(uniforms.settings.statistics_normalization_constant)), 1.0));
    } else if (uniforms.settings.output_mode == NODES_ACCESSED) {
        debug(pixel, vec4<f32>(vec3(f32(nodes_accessed) / f32(uniforms.settings.statistics_normalization_constant)), 1.0));
    } else if (uniforms.settings.output_mode == SAMPLE_STEPS) {
        debug(pixel, vec4<f32>(vec3(f32(steps_taken) / f32(uniforms.settings.statistics_normalization_constant)), 1.0));
    } else if (uniforms.settings.output_mode == DVR_PLUS_LENS_RADIUS) {
        if (pixel_radius < uniforms.settings.brick_request_radius + 0.001 && pixel_radius > uniforms.settings.brick_request_radius - 0.001) {
            color = RED;
        }
        debug(pixel, color);
    } else {
        debug(pixel, color);
    }
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}

// todo: refactor in own file, simplify
fn try_fetch_brick(ray_sample: float3, lod: u32, channel: u32, request_bricks: bool) -> Node {
    let page_table_index = compute_page_table_index(
        channel_settings[channel].page_table_index,
        lod
    );

    // todo: store volume_to_padded in page table (is it already?)
    let position_corrected = ray_sample * pt_compute_volume_to_padded(page_table_index);
    let page_address = pt_compute_page_address(page_table_index, position_corrected);
    let page = get_page(page_address);

    var requested_brick = false;
    if (page.flag == UNMAPPED) {
        if (request_bricks) {
            // todo: maybe request lower res as well?
            request_brick(int3(page_address));
            requested_brick = true;
        }
    } else if (page.flag == EMPTY) {
        return Node(
            true,
            0.0,
            true,
            float3(),
            false
        );
    } else if (page.flag == MAPPED) {
        report_usage(int3(page.location / page_directory_meta.brick_size));

        let sample_location = normalize_cache_address(
            pt_compute_cache_address(page_table_index, ray_sample, page)
        );
        return Node(
            false,
            0.0,
            true,
            sample_location,
            false
        );
    }

    return Node(
        false,
        0.0,
        false,
        float3(),
        requested_brick
    );
}

fn compute_brick_jump_distance(position: vec3<f32>, resolution_level: u32, dt: f32, t_max: f32, ray: Ray) -> f32 {
    let page_table_index = page_directory_compute_page_table_index(0, resolution_level);
    let page_subscript = pt_compute_local_page_address(page_table_index, position);
    let page_shape = pt_get_page_table_extent(page_table_index);

    let volume_to_padded = pt_compute_volume_to_padded(page_table_index);

    let grid_ray = make_grid_ray(
        Ray (ray.origin, ray.direction * pt_compute_volume_to_padded(page_table_index), t_max),
        t_max
    );

    // when going the positive direction, we want to check the node bound's max. axis, otherwise we'll use the
    // node bound's min. axis, i.e., the axis we already have in the subscript
    let next_axis_indices = vec3<i32>(page_subscript) + grid_ray.grid_step;

    // to compute the distance we can jump over, we first compute the normalized floating point coordinates of
    // the next axes.
    // note: we don't use `subscript_to_normalized_address` here because we might need values outside the unit cube
    let next_axis_coords = vec3<f32>(next_axis_indices) / vec3<f32>(page_shape);

    return compute_jump_distance(position, next_axis_coords, dt, grid_ray);
}
