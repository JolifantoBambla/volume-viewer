use std::{
    borrow::Cow,
    sync::Arc,
};
use wgpu::{
    BindGroup,
    BindGroupEntry,
    BindGroupLayout,
    Label,
    TextureView,
};
use crate::renderer::{
    context::GPUContext,
    pass::{GPUPass, AsBindGroupEntries},
};

pub struct PresentToScreenBindGroup {

}

impl AsBindGroupEntries for PresentToScreenBindGroup {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        Vec::new()
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("present_to_screen.wgsl"))),
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

    pub fn encode(&self, bind_group: &BindGroup, command_encoder: &mut wgpu::CommandEncoder, view: &TextureView) {
        let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0
                    }),
                    store: true
                }
            }],
            depth_stencil_attachment: None
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..6, 0..1);
    }
}

impl GPUPass<PresentToScreenBindGroup> for PresentToScreen {
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
