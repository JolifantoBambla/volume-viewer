/// The base for all DVR passes

// INCLUDES -----------------------------------------------------------------------------------------------------------
// COMMON
@include(aabb)
@include(camera)
@include(channel_settings)
@include(constant)
@include(grid_leap)
@include(lighting)
@include(output_modes)
@include(page_table)
@include(ray)
@include(timestamp)
@include(transform)
@include(type_alias)

// includes that require the shader to define certain functions
@include(volume_util)

// Note: volume_accelerator needs to map to some file!
@include(volume_accelerator)

// EXTRA
@include(extra_includes)
// e.g., octree specific: @include(page_directory_meta_util)


// BINDINGS ------------------------------------------------------------------------------------------------------------
// all DVR passes share bind group 0 and 1 but may specify more

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var volume_sampler: sampler;
@group(0) @binding(2) var result: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage> channel_settings: array<ChannelSettings>;

@group(1) @binding(0) var<uniform> page_directory_meta: PageDirectoryMeta;
@group(1) @binding(1) var<storage> page_table_meta: PageTableMetas;
@group(1) @binding(2) var page_directory: texture_3d<u32>;
@group(1) @binding(3) var brick_cache: texture_3d<f32>;
@group(1) @binding(4) var brick_usage_buffer: texture_storage_3d<r32uint, write>;
@group(1) @binding(5) var request_buffer: texture_storage_3d<r32uint, write>;

@include(extra_bind_groups)


// HELPER --------------------------------------------------------------------------------------------------------------

fn store_result(pixel: vec2<i32>, color: vec4<f32>) {
    textureStore(result, pixel, color);
}

fn sample_volume(x: vec3<f32>) -> f32 {
    return f32(textureSampleLevel(brick_cache, volume_sampler, x, 0.).x);
}

fn load_voxel(x: vec3<u32>) -> f32 {
    return f32(textureLoad(brick_cache, vec3<i32>(x), 0).x);
}

/// Reports usage of a cached brick by storing the current frame's timestamp in the `brick_usage_buffer`.
///
/// # Arguments
/// * `brick_address`: the address of the cached brick. The actual brick is located at `brick_address` * `brick_size`
fn report_usage(brick_address: vec3<i32>) {
    textureStore(brick_usage_buffer, brick_address, vec3<u32>(uniforms.timestamp));
}

/// Requests a brick missing in the `brick_cache` by storing the current frame's timestamp in the `request_buffer`.
///
/// # Arguments
/// * `page_address`:
fn request_brick(page_address: vec3<i32>) {
    textureStore(request_buffer, page_address, vec3<u32>(uniforms.timestamp));
}

fn normalize_cache_address(cache_address: vec3<u32>) -> vec3<f32> {
    let cache_size = vec3<f32>(textureDimensions(brick_cache));
    return saturate(vec3<f32>(cache_address) / cache_size);
}

fn get_page(page_address: vec3<u32>) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, vec3<i32>(page_address), 0));
}

fn compute_page_table_index(channel: u32, lod: u32) -> u32 {
    return channel + lod * page_directory_meta.max_channels;
}

fn is_saturated(color: vec4<f32>) -> bool {
    return color.a > 0.95;
}

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // terminate thread if it's outside the window bounds
    let window_size = vec2<u32>(textureDimensions(result));
    if (any(window_size < global_id.xy)) {
        return;
    }

    let pixel = vec2<i32>(global_id.xy);
    let resolution = vec2<f32>(window_size);

    // initialize result
    store_result(pixel, vec4<f32>());

    let center_to_pixel = length(vec2<f32>(pixel) - (resolution / 2.0));
    let center_to_corner = length(resolution / 2.0);
    let request_bricks = center_to_pixel / center_to_corner <= uniforms.settings.brick_request_radius;

    // generate a ray and transform it to the volume's space (i.e. where the volume is a unit cube with x in [0,1]^3)
    let ray_ws = generate_camera_ray(uniforms.camera, vec2<f32>(pixel), resolution);
    let ray_os = transform_ray(ray_ws, uniforms.volume_transform.world_to_object);

    // terminate thread if the ray isn't within the unit cube
    let volume_bounds_os = AABB(vec3<f32>(0.), vec3<f32>(1.));
    let intersection_record = intersect_aabb(ray_os, volume_bounds_os);
    if (!intersection_record.hit) {
        return;
    }

    let object_to_view = uniforms.camera.transform.world_to_object * uniforms.volume_transform.object_to_world;

    let t_min = max(0., intersection_record.t_min);
    let t_max = intersection_record.t_max;

    let num_visible_channels = uniforms.settings.num_visible_channels;
    let num_resolutions = page_directory_meta.max_resolutions;
    let max_num_channels = page_directory_meta.max_channels;

    let inv_high_res_size = 1. / vec3<f32>(page_table_meta.metas[0].volume_size);
    let brick_size = page_table_meta.metas[0].brick_size;

    let lowest_res = min(channel_settings[0].min_lod, num_resolutions - 1);
    let highest_res = min(channel_settings[0].max_lod, lowest_res);
    let lod_factor = channel_settings[0].lod_factor * max_component(inv_high_res_size);

    // Set up state tracking
    var color = vec4<f32>();
    var requested_brick = !request_bricks;
    var last_lod = 0u;
    var steps_taken = 0u;
    var bricks_accessed = 0u;
    var nodes_accessed = 0u;

    let offset = hash_u32_to_f32(pcg_hash(u32(pixel.x + pixel.x * pixel.y)));
    let dt_scale = uniforms.settings.step_scale;

    // todo: add stuff from other branch
    var dt_vec = 1. / (vec3<f32>(page_table_meta.metas[last_lod].volume_size) * abs(ray_os.direction));
    var dt = dt_scale * min_component(dt_vec);
    var p = ray_at(ray_os, t_min + offset * dt);

    let first_channel_index = channel_settings[0].page_table_index;

    let grid_ray = make_grid_ray(ray_os, t_max);
    let start_subdivision_index = 2u; // todo: add to global settings and add slider
    var last_subdivision_level = start_subdivision_index;

    for (var t = t_min; t < t_max; t += dt) {
        let distance_to_camera = abs((object_to_view * vec4<f32>(p, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res, lod_factor);
        let lod_changed = lod != last_lod;
        if (lod_changed) {
            last_lod = lod;
            // todo: add stuff from other branch
            dt_vec = 1. / (vec3<f32>(page_table_meta.metas[compute_page_table_index(first_channel_index, lod)].volume_size) * abs(ray_os.direction));
            dt = dt_scale * min_component(dt_vec);
        }

        var subdivision_index = last_subdivision_level;

        @include(dvr_traversal_algorithm)

        steps_taken += 1;
        if (steps_taken >= uniforms.settings.max_steps) {
            break;
        }
        if (is_saturated(color)) {
            break;
        }

        if (empty_channels == num_visible_channels) {
            @include(dt_jump)
            p += ray_os.direction * dt_jump;
            t += dt_jump;
        }

        p += ray_os.direction * dt;
    }

    // todo: add stuff from other branch
    if (uniforms.settings.output_mode == DVR) {
        store_result(pixel, color);
    } else if (uniforms.settings.output_mode == BRICKS_ACCESSED) {
        store_result(pixel, vec4<f32>(vec3(f32(bricks_accessed) / 255.0), 1.0));
    } else if (uniforms.settings.output_mode == NODES_ACCESSED) {
        store_result(pixel, vec4<f32>(vec3(f32(nodes_accessed) / 255.0), 1.0));
    } else if (uniforms.settings.output_mode == SAMPLE_STEPS) {
        store_result(pixel, vec4<f32>(vec3(f32(steps_taken) / 255.0), 1.0));
    }
}

struct Brick {
    sample_address: vec3<f32>,
    was_requested: bool,
    is_mapped: bool,
    is_empty: bool
}

fn try_fetch_brick(ray_sample: vec3<f32>, lod: u32, channel: u32, request_bricks: bool) -> Brick {
    let page_table_index = compute_page_table_index(channel_settings[channel].page_table_index, lod);

    // todo: store volume_to_padded in page table (is it already?)
    let position_corrected = ray_sample * pt_compute_volume_to_padded(page_table_index);
    let page_address = pt_compute_page_address(page_table_index, position_corrected);
    let page = get_page(page_address);

    var brick = Brick();
    brick.is_mapped = page.flag == MAPPED;
    brick.is_empty = page.flag == EMPTY;
    brick.was_requested = page.flag == UNMAPPED && request_bricks;
    if (brick.was_requested) {
        request_brick(vec3<i32>(page_address));
    } else if (brick.is_mapped) {
        report_usage(vec3<i32>(page.location / page_directory_meta.brick_size));
        brick.sample_address = normalize_cache_address(pt_compute_cache_address(page_table_index, ray_sample, page));
    }
    return brick;
}
