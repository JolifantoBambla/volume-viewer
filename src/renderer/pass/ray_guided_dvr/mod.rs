use crate::renderer::camera::TransformUniform;
use crate::renderer::{
    camera::CameraUniform,
    context::GPUContext,
    pass::{AsBindGroupEntries, GPUPass},
};
use crate::{MultiChannelVolumeRendererSettings, SparseResidencyTexture3D};
use glam::{UVec4, Vec4};
use std::{borrow::Cow, sync::Arc};
use bytemuck::Contiguous;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout};
use wgsl_preprocessor::WGSLPreprocessor;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChannelSettings {
    pub color: Vec4,
    pub channel_index: u32,
    pub max_lod: u32,
    pub min_lod: u32,
    pub threshold_lower: f32,
    pub threshold_upper: f32,
    pub visible: u32,
    pub page_table_index: u32,
    padding2: u32,
}

impl ChannelSettings {
    pub fn from_channel_settings_with_mapping(settings: &crate::renderer::settings::ChannelSettings, mapping: Vec<Option<usize>>) -> Self {
        let mut s = Self::from(settings);
        //if s.visible {
            //mapping.in
            //s.page_table_index = mapping[]
        //}
        s
    }
}

impl From<&crate::renderer::settings::ChannelSettings> for ChannelSettings {
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
            padding2: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalSettings {
    pub render_mode: u32,
    pub step_scale: f32,
    pub max_steps: u32,
    pub num_visible_channels: u32,
    pub background_color: Vec4,
}

impl From<&MultiChannelVolumeRendererSettings> for GlobalSettings {
    fn from(settings: &MultiChannelVolumeRendererSettings) -> Self {
        Self {
            render_mode: settings.render_mode as u32,
            step_scale: settings.step_scale,
            max_steps: settings.max_steps,
            num_visible_channels: settings.channel_settings.iter().filter(|c| c.visible).count() as u32,
            background_color: Vec4::from(settings.background_color),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

pub struct Resources<'a> {
    pub volume_sampler: &'a wgpu::Sampler,
    pub output: &'a wgpu::TextureView,
    pub uniforms: &'a wgpu::Buffer,
    pub channel_settings: &'a wgpu::Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(self.volume_sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(self.output),
            },
            BindGroupEntry {
                binding: 3,
                resource: self.channel_settings.as_entire_binding(),
            },
        ]
    }
}

pub struct RayGuidedDVR {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
}

impl RayGuidedDVR {
    pub fn new(
        volume_texture: &SparseResidencyTexture3D,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("ray_cast.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let internal_bind_group_layout = pipeline.get_bind_group_layout(1);
        let internal_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &internal_bind_group_layout,
            entries: &volume_texture.as_bind_group_entries(),
        });

        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
            internal_bind_group,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        bind_group: &BindGroup,
        output_extent: &wgpu::Extent3d,
    ) {
        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ray Guided DVR"),
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, bind_group, &[]);
        cpass.set_bind_group(1, &self.internal_bind_group, &[]);
        cpass.insert_debug_marker(self.label());
        cpass.dispatch_workgroups(
            (output_extent.width as f32 / 16.).ceil() as u32,
            (output_extent.height as f32 / 16.).ceil() as u32,
            1,
        );
    }
}

impl<'a> GPUPass<Resources<'a>> for RayGuidedDVR {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "Ray Guided DVR"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
