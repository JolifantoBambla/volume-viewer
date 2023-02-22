use crate::app::renderer::dvr::Resources;
use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
use crate::resource::VolumeManager;
#[cfg(feature = "timestamp-query")]
use crate::timing::timestamp_query_helper::TimestampQueryHelper;
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupLayout};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

#[derive(Debug)]
pub struct PageTableDVR {
    ctx: Arc<Gpu>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: BindGroupLayout,
    internal_bind_group: BindGroup,
}

impl PageTableDVR {
    pub fn new(
        volume_texture: &VolumeManager,
        wgsl_preprocessor_base: &WGSLPreprocessor,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let mut wgsl_preprocessor = wgsl_preprocessor_base.clone();
        wgsl_preprocessor.include(
            "volume_accelerator",
            include_str!("page_table_volume_accessor.wgsl"),
        );

        let shader_module = ctx
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
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader_module,
                entry_point: "main",
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let internal_bind_group_layout = pipeline.get_bind_group_layout(1);
        let internal_bind_group = ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
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
        #[cfg(feature = "timestamp-query")] timestamp_query_helper: &mut TimestampQueryHelper,
    ) {
        #[cfg(feature = "timestamp-query")]
        let timestamp_writes = timestamp_query_helper.make_compute_pass_timestamp_write_pair();
        #[cfg(not(feature = "timestamp-query"))]
        let timestamp_writes = Vec::new();

        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Ray Guided DVR"),
            timestamp_writes: timestamp_writes.as_slice(),
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

impl<'a> GPUPass<Resources<'a>> for PageTableDVR {
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
