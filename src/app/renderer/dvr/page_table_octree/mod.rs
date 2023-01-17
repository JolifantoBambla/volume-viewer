use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::VolumeManager;
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

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

#[derive(Debug)]
pub struct PageTableOctreeDVR {
    ctx: Arc<Gpu>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
}

impl PageTableOctreeDVR {
    pub fn new(
        volume_texture: &VolumeManager,
        wgsl_preprocessor_base: &WGSLPreprocessor,
        gpu: &Arc<Gpu>,
    ) -> Self {
        let mut wgsl_preprocessor = wgsl_preprocessor_base.clone();
        wgsl_preprocessor.include(
            "volume_accelerator",
            include_str!("page_table_octree_volume_accessor.wgsl"),
        );

        let shader_module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("../ray_cast.wgsl"))
                        .ok()
                        .unwrap(),
                )),
            });
        let pipeline = gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let internal_bind_group_layout = pipeline.get_bind_group_layout(1);
        let internal_bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &internal_bind_group_layout,
            entries: &volume_texture.as_bind_group_entries(),
        });

        Self {
            ctx: gpu.clone(),
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

impl<'a> GPUPass<Resources<'a>> for PageTableOctreeDVR {
    fn ctx(&self) -> &Arc<Gpu> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "Ray Guided DVR"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
