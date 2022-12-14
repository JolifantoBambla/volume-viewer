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
@include(page_table_volume_accessor)

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
    lod_factor: f32,
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

fn clone_page_table_meta(level: u32) -> PageTableMeta {
    let page_table_meta = page_table_meta.metas[level];
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
        channel_settings.page_table_index,
        channel_settings.lod_factor
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
    let resolution = page_table_meta.metas[level];
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

    let num_channels = uniforms.settings.num_visible_channels;
    let num_resolutions = page_directory_meta.max_resolutions;

    let cs = clone_channel_settings(0);

    let inv_high_res_size = 1. / float3(page_table_meta.metas[0].volume_size);
    let brick_size = page_table_meta.metas[0].brick_size;

    let lowest_res = min(cs.min_lod, num_resolutions - 1);
    let highest_res = min(cs.max_lod, lowest_res);
    let lod_factor = cs.lod_factor * max_component(inv_high_res_size);

    // Set up state tracking
    var color = float4();
    var requested_brick = false;
    var last_lod = 0u;
    var steps_taken = 0u;

    let offset = hash_u32_to_f32(pcg_hash(u32(pixel.x + 640 * pixel.y)));
    let dt_scale = uniforms.settings.step_scale;

    var dt_vec = 1. / (float3(page_table_meta.metas[last_lod].volume_size) * abs(ray_os.direction));
    var dt = dt_scale * min_component(dt_vec);
    var p = ray_at(ray_os, t_min + offset * dt);

    let first_channel_index = channel_settings_list.channels[0].page_table_index;

    for (var t = t_min; t < t_max; t += dt) {
        let distance_to_camera = abs((object_to_view * float4(p, 1.)).z);
        let lod = select_level_of_detail(distance_to_camera, highest_res, lowest_res, lod_factor);
        let lod_changed = lod != last_lod;
        if (lod_changed) {
            last_lod = lod;
            dt_vec = 1. / (float3(page_table_meta.metas[compute_page_table_index(first_channel_index, lod)].volume_size) * abs(ray_os.direction));
            dt = dt_scale * min_component(dt_vec);
        }

        for (var channel = 0u; channel < num_channels; channel += 1u) {
            let node = va_get_node(p, lod, channel, !requested_brick);
            requested_brick = requested_brick || node.requested_brick;
            if (node.is_mapped) {
                let value = sample_volume(node.sample_address);

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
        if (is_saturated(color)) {
            break;
        }

        p += ray_os.direction * dt;
    }

    debug(pixel, color);
}

fn is_saturated(color: float4) -> bool {
    // todo: make threshold configurable
    return color.a > 0.95;
}
