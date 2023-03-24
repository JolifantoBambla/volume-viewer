use crate::app::renderer::common::{CameraUniform, TransformUniform};
use crate::MultiChannelVolumeRendererSettings;
use bytemuck::Contiguous;
use glam::{UVec3, UVec4, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuChannelSettings {
    pub color: Vec4,
    pub channel_index: u32,
    pub max_lod: u32,
    pub min_lod: u32,
    pub threshold_lower: f32,
    pub threshold_upper: f32,
    pub visible: u32,
    pub page_table_index: u32,
    pub lod_factor: f32,
}

impl From<&crate::renderer::settings::ChannelSettings> for GpuChannelSettings {
    fn from(settings: &crate::renderer::settings::ChannelSettings) -> Self {
        Self {
            color: Vec4::from(settings.color),
            channel_index: settings.channel_index,
            max_lod: settings.max_lod,
            min_lod: settings.min_lod,
            threshold_lower: settings.threshold_lower,
            threshold_upper: settings.threshold_upper,
            visible: settings.visible as u32,
            page_table_index: u32::MAX_VALUE,
            lod_factor: settings.lod_factor,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalSettings {
    pub render_mode: u32,
    pub step_scale: f32,
    pub max_steps: u32,
    pub num_visible_channels: u32,
    pub background_color: Vec4,
    pub output_mode: u32,
    pub padding1: UVec3,
    pub padding2: Vec4,
}

impl From<&MultiChannelVolumeRendererSettings> for GlobalSettings {
    fn from(settings: &MultiChannelVolumeRendererSettings) -> Self {
        Self {
            render_mode: settings.render_mode as u32,
            step_scale: settings.step_scale,
            max_steps: settings.max_steps,
            num_visible_channels: settings
                .channel_settings
                .iter()
                .filter(|c| c.visible)
                .count() as u32,
            background_color: Vec4::from(settings.background_color),
            output_mode: settings.output_mode as u32,
            padding1: UVec3::ZERO,
            padding2: Vec4::ZERO,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub camera: CameraUniform,
    pub volume_transform: TransformUniform,
    pub timestamp: UVec4,
    pub settings: GlobalSettings,
}

impl Uniforms {
    pub fn new(
        camera: CameraUniform,
        object_to_world: glam::Mat4,
        timestamp: u32,
        settings: &MultiChannelVolumeRendererSettings,
    ) -> Self {
        let volume_transform = TransformUniform::from_object_to_world(object_to_world);
        Self {
            camera,
            volume_transform,
            timestamp: UVec4::new(timestamp, timestamp, timestamp, timestamp),
            settings: GlobalSettings::from(settings),
        }
    }
}
