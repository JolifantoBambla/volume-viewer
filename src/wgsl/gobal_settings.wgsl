@include(camera)
@include(transform)

struct GlobalSettings {
    render_mode: u32,
    step_scale: f32,
    max_steps: u32,
    num_visible_channels: u32,
    background_color: vec4<f32>,
    output_mode: u32,
    statistics_normalization_constant: u32,
    padding1: vec2<u32>,
    voxel_spacing: vec3<f32>,
    brick_request_radius: f32,
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
