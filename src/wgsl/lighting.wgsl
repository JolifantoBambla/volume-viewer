// This interface requires the following identifiers to be defined:
// bindings:
//  - channel_settings: array<ChannelSettings>;

@include(channel_settings)

fn compute_lighting(voxel_value: f32, channel: u32, sampling_frequency_factor: f32, color: ptr<function, vec4<f32>>) {
    let lower_bound = channel_settings[channel].threshold_lower;
    let upper_bound = channel_settings[channel].threshold_upper;
    if (voxel_value >= lower_bound && voxel_value <= upper_bound) {
        // todo: transfer functions
        let transfer_function_sample = channel_settings[channel].color;
        var voxel_color = float4(transfer_function_sample.rgb, voxel_value * transfer_function_sample.a);
        voxel_color.a = 1.0 - pow(1.0 - voxel_color.a, sampling_frequency_factor);

        let transparency = (1.0 - (*color).a) * voxel_color.a;
        *color += vec4<f32>(transparency * voxel_color.rgb, transparency);
    }
}
