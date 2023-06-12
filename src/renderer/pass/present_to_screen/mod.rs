use crate::renderer::pass::{AsBindGroupEntries, GPUPass};
#[cfg(feature = "timestamp-query")]
use crate::timing::timestamp_query_helper::TimestampQueryHelper;
use std::{borrow::Cow, sync::Arc};
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, Buffer, CommandEncoder, RenderPipeline, SurfaceConfiguration, TextureView};
use wgpu_framework::context::Gpu;

pub struct Resources<'a> {
    pub sampler: &'a wgpu::Sampler,
    pub source_texture: &'a TextureView,
    pub background_color: &'a Buffer,
}

impl<'a> AsBindGroupEntries for Resources<'a> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Sampler(self.sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(self.source_texture),
            },
            BindGroupEntry {
                binding: 2,
                resource: self.background_color.as_entire_binding(),
            }
        ]
    }
}

#[derive(Debug)]
pub struct PresentToScreen {
    ctx: Arc<Gpu>,
    pipeline: RenderPipeline,
    bind_group_layout: BindGroupLayout,
}

impl PresentToScreen {
    pub fn new(ctx: &Arc<Gpu>, surface_configuration: &SurfaceConfiguration) -> Self {
        let shader_module = ctx
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "present_to_screen.wgsl"
                ))),
            });
        let pipeline = ctx
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader_module,
                    entry_point: "vert_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: "frag_main",
                    targets: &[Some(surface_configuration.format.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
            });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        Self {
            ctx: ctx.clone(),
            pipeline,
            bind_group_layout,
        }
    }

    pub fn encode(
        &self,
        command_encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        view: &TextureView,
        #[cfg(feature = "timestamp-query")] timestamp_query_helper: &mut TimestampQueryHelper,
    ) {
        #[cfg(feature = "timestamp-query")]
        let timestamp_writes = Some(timestamp_query_helper.make_render_pass_timestamp_writes());
        #[cfg(not(feature = "timestamp-query"))]
        let timestamp_writes = None;

        let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        rpass.insert_debug_marker(self.label());
        rpass.draw(0..6, 0..1);
    }
}

impl<'a> GPUPass<Resources<'a>> for PresentToScreen {
    fn ctx(&self) -> &Arc<Gpu> {
        &self.ctx
    }

    fn label(&self) -> &str {
        "PresentToScreen"
    }

    fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}
