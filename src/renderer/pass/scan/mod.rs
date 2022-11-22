use std::borrow::Cow;
use std::sync::Arc;
use wgpu::{BindGroup, ComputePipeline, ComputePipelineDescriptor, Label, ShaderModuleDescriptor, ShaderSource};
use wgsl_preprocessor::WGSLPreprocessor;
use crate::renderer::context::GPUContext;

struct Scan {
    ctx: Arc<GPUContext>,
    scan_pipeline: ComputePipeline,
    sum_pipeline: ComputePipeline,
    scan_bind_group: BindGroup,
    sum_bind_group: BindGroup,
}

impl Scan {
    pub fn new(
        num_elements: u32,
        use_max_input: u32,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        let scan_shader_module = ctx
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Label::from("Scan"),
                source: ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("scan.wgsl"))
                        .ok()
                        .unwrap()
                )),
            });
        let scan_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Label::from("Scan"),
                layout: None,
                module: &scan_shader_module,
                entry_point: "main",
            });
        let scan_bind_group_layout = scan_pipeline.get_bind_group_layout(0);


        let sum_shader_module = ctx
            .device
            .create_shader_module(ShaderModuleDescriptor {
                label: Label::from("Sum"),
                source: ShaderSource::Wgsl(Cow::Borrowed(
                    &*wgsl_preprocessor
                        .preprocess(include_str!("add.wgsl"))
                        .ok()
                        .unwrap()
                )),
            });
        let sum_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Label::from("Sum"),
                layout: None,
                module: &sum_shader_module,
                entry_point: "main",
            });
        let sum_bind_group_layout = scan_pipeline.get_bind_group_layout(0);

        todo!();
        Self {
            ctx: ctx.clone(),
            scan_pipeline,
            sum_pipeline,
            scan_bind_group: (),
            sum_bind_group: (),
        }
    }
}