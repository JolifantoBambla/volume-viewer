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
@include(ray)
@include(timestamp)
@include(transform)
@include(type_alias)

// includes that require the shader to define certain functions
@include(volume_util)

// Note: volume_accelerator needs to map to some file!
@include(volume_accelerator)

@include(multichannel_octree_util)
@include(octree_node)
@include(octree_node_util)
@include(page_directory_meta_util)
@include(volume_subdivision)
@include(volume_subdivision_util)

fn debug(pixel: int2, color: float4) {
    textureStore(result, pixel, color);
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
@group(1) @binding(6) var<storage> volume_subdivisions: array<VolumeSubdivision>;
@group(1) @binding(7) var<storage> octree_nodes: array<u32>;

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

// todo: do I really need this?
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
    let cache_size = uint3(textureDimensions(brick_cache));
    return subscript_to_normalized_address(cache_address, cache_size);
}

// todo: add multiple virtualization levels
fn get_page(page_address: uint3) -> PageTableEntry {
    return to_page_table_entry(textureLoad(page_directory, int3(page_address), 0));
}

// todo: I'm pretty sure there is a util function in some other file for this
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
    let max_num_channels = page_directory_meta.max_channels;

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

    // todo: remove (debug)
    var last_node_index = 4294967295u;

    let grid_ray = make_grid_ray(ray_os, t_max);
    let start_subdivision_index = 2u;
    var last_subdivision_level = start_subdivision_index;

    for (var t = t_min; t < t_max; t += dt) {
        let distance_to_camera = abs((object_to_view * float4(p, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res, lod_factor);
        let lod_changed = lod != last_lod;
        if (lod_changed) {
            last_lod = lod;
            dt_vec = 1. / ((float3(page_table_meta.metas[compute_page_table_index(first_channel_index, lod)].volume_size) * uniforms.settings.voxel_spacing) * abs(ray_os.direction));
            dt = dt_scale * min_component(dt_vec);
        }

        // todo: pass this as a channel setting
        let homogeneous_threshold = 1.0;

        // todo: compute this from some params (maybe just use lod selection mechanism?)
        let target_culling_level = subdivision_get_leaf_node_level_index();

        var channel = 0u;
        // todo: make start level configurable (e.g., start at level 2 because 0 and 1 are unlikely to be empty anyway)
        var subdivision_index = last_subdivision_level;
        var empty_channels = 0u;
        var homogeneous_channels = 0u;

        // todo: remove (debug)
        var terminated_at_index = 0u;

        var last_channel = channel;
        var lower_threshold = u32(floor((channel_settings[channel].threshold_lower - EPSILON) * 255.0));
        var upper_threshold = u32(floor((channel_settings[channel].threshold_upper + EPSILON) * 255.0));
        var channel_lod_factor = channel_settings[channel].lod_factor * max_component(inv_high_res_size);

        while (subdivision_index <= target_culling_level && channel < num_visible_channels) {
            nodes_accessed += 1;
            if (last_channel != channel) {
                lower_threshold = u32(floor((channel_settings[channel].threshold_lower - EPSILON) * 255.0));
                upper_threshold = u32(floor((channel_settings[channel].threshold_upper + EPSILON) * 255.0));
                channel_lod_factor = channel_settings[channel].lod_factor * max_component(inv_high_res_size);
            }

            let multichannel_global_node_index = to_multichannel_node_index(
                subdivision_idx_compute_node_index(subdivision_index, p),
                max_num_channels,
                page_directory_meta_get_channel_index(channel_settings[channel].page_table_index)
            );

            terminated_at_index = multichannel_global_node_index;

            let node = node_idx_load_global(multichannel_global_node_index);
            if (node_has_no_data(node)) {
                if (!requested_brick) {
                    pt_request_brick(p, channel_settings[channel].min_lod, channel);
                    requested_brick = true;
                }
                channel += 1;
                continue;
            }

            if (node_is_empty(node, lower_threshold, upper_threshold)) {
                channel += 1;
                empty_channels += 1;
                continue;
            }
            /*
            if (node_is_homogeneous(node, homogeneous_threshold)) {
                let node_value = f32(node_get_min(node)) / 255.0;
                color += compute_lighting(node_value, channel, dt_scale, color);
                channel += 1;
                homogeneous_channels += 1;
                continue;
            }
            */
            if (node_is_not_mapped(node)) {
                if (!requested_brick) {
                    pt_request_brick(p, channel_settings[channel].min_lod, channel);
                    requested_brick = true;
                }
                channel += 1;
                continue;
            }
            if (subdivision_index != target_culling_level) {
                subdivision_index += 1;
                continue;
            }

            let target_resolution = va_compute_lod(distance_to_camera, channel, channel_lod_factor);
            let target_mask = node_make_mask_for_resolution(target_resolution);
            let resolution_mapping = node_get_partially_mapped_resolutions(node);
            let target_partially_mapped = (target_mask & resolution_mapping) > 0;
            var value = -1.0;
            if (target_partially_mapped) {
                let brick = try_fetch_brick(p, target_resolution, channel, !requested_brick);
                requested_brick = requested_brick || brick.requested_brick;
                if (brick.is_mapped) {
                    bricks_accessed += 1;
                    value = sample_volume(brick.sample_address);
                }
            }
            if (value < 0.0) {
                if (!requested_brick) {
                    pt_request_brick(p, target_resolution, channel);
                    requested_brick = true;
                }
                // todo: use countLeadingZeros & countTrailingZeros here
                let channel_lowest_lod = channel_settings[channel].min_lod;
                for (var res = target_resolution + 1; res <= channel_lowest_lod; res += 1) {
                    let brick = try_fetch_brick(p, res, channel, false);
                    if (brick.is_mapped) {
                        bricks_accessed += 1;
                        value = sample_volume(brick.sample_address);
                        break;
                    }
                }
                if (value < 0.0) {
                    let channel_highest_lod = channel_settings[channel].max_lod;
                    for (var res = target_resolution - 1; res >= channel_highest_lod; res -= 1) {
                        let brick = try_fetch_brick(p, res, channel, false);
                        if (brick.is_mapped) {
                            bricks_accessed += 1;
                            value = sample_volume(brick.sample_address);
                            break;
                        }
                    }
                }
            }

            compute_lighting(value, channel, dt_scale, &color);

            channel += 1;
        }

        steps_taken += 1;
        if (steps_taken >= uniforms.settings.max_steps) {
            break;
        }
        if (is_saturated(color)) {
            break;
        }

        if (empty_channels == num_visible_channels) {
            /*
            if (terminated_at_index == last_node_index) {
                //color = RED;
                //break;
            }
            */

            let dt_jump = compute_node_jump_distance(p, subdivision_index, dt, grid_ray);
            p += ray_os.direction * dt_jump;
            t += dt_jump;
        }

        p += ray_os.direction * dt;
        last_subdivision_level = max(start_subdivision_index, subdivision_index - 1);

        // todo: remove(debug)
        last_node_index = terminated_at_index;
    }

    if (steps_taken <= 3u) {
        //color = YELLOW;
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

fn pt_request_brick(ray_sample: float3, lod: u32, channel: u32) {
    let page_table_index = compute_page_table_index(
        channel_settings[channel].page_table_index,
        lod
    );

    // todo: store volume_to_padded in page table (is it already?)
    let position_corrected = ray_sample * pt_compute_volume_to_padded(page_table_index);
    let page_address = pt_compute_page_address(page_table_index, position_corrected);
    request_brick(int3(page_address));
}

// todo: simplify Node / Brick
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
            request_brick(int3(page_address));
            requested_brick = true;
        }
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

fn compute_node_jump_distance(position: vec3<f32>, subdivision_index: u32, dt: f32, grid_ray: GridRay) -> f32 {
    let subdivision_shape = vec3<f32>(subdivision_idx_get_shape(subdivision_index));
    let node_subscript = subdivision_idx_compute_subscript(subdivision_index, position);

    // when going the positive direction, we want to check the node bound's max. axis, otherwise we'll use the
    // node bound's min. axis, i.e., the axis we already have in the subscript
    let next_axis_indices = vec3<i32>(node_subscript) + grid_ray.grid_step;

    // to compute the distance we can jump over, we first compute the normalized floating point coordinates of
    // the next axes.
    // note: we don't use `subscript_to_normalized_address` here because we might need values outside the unit cube
    let next_axis_coords = vec3<f32>(next_axis_indices) / subdivision_shape;

    return compute_jump_distance(position, next_axis_coords, dt, grid_ray);
}
