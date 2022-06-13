use std::borrow::Cow;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, CommandEncoder, Label};
use crate::renderer::context::GPUContext;

pub trait AsBindGroupEntries {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry>;
}

pub trait GPUPass<'a, T: AsBindGroupEntries> {
    // todo: maybe move out of pub trait and use in other resources as well
    fn ctx(&self) -> &Arc<GPUContext>;
    fn label(&self) -> wgpu::Label;

    fn bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    fn create_bind_group(&self, resources: T) -> wgpu::BindGroup {
        self.ctx().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.bind_group_layout(),
            entries: &resources.as_bind_group_entries()
        })
    }
}

pub struct PresentToScreenBindGroupBuilder {

}

impl<'a> Into<[wgpu::BindGroupEntry<'a>]> for PresentToScreenBindGroup {
    fn into(self) -> [BindGroupEntry<'a>] {
        todo!()
    }
}

pub struct PresentToScreen {
    ctx: Arc<GPUContext>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl PresentToScreen {
    pub fn new(ctx: &Arc<GPUContext>) -> Self {
        let shader_module = ctx.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("full_screen_quad.wgsl"))),
        });
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: "vert_main",
                buffers: &[]
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: "frag_main",
                targets: &[ctx.surface_configuration.as_ref().unwrap().format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
        }
    }
}

impl<'a> GPUPass<'a, PresentToScreenBindGroup> for PresentToScreen {
    fn ctx(&self) -> &Arc<GPUContext> {
        &self.ctx
    }

    fn label(&self) -> Label {
        Some("PresentToScreen")
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}


