use crate::renderer::pass::{ComputePipelineData, StaticComputeEncodeDescriptor};
use crate::resource::buffer::{map_buffer, TypedBuffer};
use crate::resource::MappableBuffer;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferUsages, CommandEncoder, CommandEncoderDescriptor,
    ComputePass, ComputePassDescriptor, ComputePipelineDescriptor, Device, Label,
    ShaderModuleDescriptor, ShaderSource,
};
use wgpu_framework::context::Gpu;
use wgsl_preprocessor::WGSLPreprocessor;

const WORKGROUP_SIZE: u32 = 256;
const WORKGROUP_SIZE_DOUBLED: u32 = WORKGROUP_SIZE * 2;

#[derive(Debug)]
pub struct Scan {
    passes: Vec<StaticComputeEncodeDescriptor>,
}

impl Scan {
    pub fn new(
        input_buffer: &TypedBuffer<u32>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<Gpu>,
    ) -> Self {
        if !input_buffer.supports(BufferUsages::STORAGE) {
            panic!("buffer to scan needs to support STORAGE");
        }

        let scan_shader_module = ctx.device().create_shader_module(ShaderModuleDescriptor {
            label: Label::from("Sum"),
            source: ShaderSource::Wgsl(Cow::Borrowed(
                &*wgsl_preprocessor
                    .preprocess(include_str!("scan.wgsl"))
                    .ok()
                    .unwrap(),
            )),
        });
        let scan_pipeline = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("Scan"),
                layout: None,
                module: &scan_shader_module,
                entry_point: "main",
            },
            ctx.device(),
        );

        let sum_shader_module = ctx.device().create_shader_module(ShaderModuleDescriptor {
            label: Label::from("Sum"),
            source: ShaderSource::Wgsl(Cow::Borrowed(
                &*wgsl_preprocessor
                    .preprocess(include_str!("add.wgsl"))
                    .ok()
                    .unwrap(),
            )),
        });
        let sum_pipeline = ComputePipelineData::new(
            &ComputePipelineDescriptor {
                label: Label::from("Sum"),
                layout: None,
                module: &sum_shader_module,
                entry_point: "main",
            },
            ctx.device(),
        );

        let mut passes = Vec::new();
        Scan::create_recursive_bind_groups(
            input_buffer,
            &scan_pipeline,
            &sum_pipeline,
            &mut passes,
            ctx.device(),
        );

        Self { passes }
    }

    fn create_recursive_bind_groups(
        input: &TypedBuffer<u32>,
        scan_pipeline: &ComputePipelineData<1>,
        sum_pipeline: &ComputePipelineData<1>,
        passes: &mut Vec<StaticComputeEncodeDescriptor>,
        device: &Device,
    ) {
        let reduced_size =
            f32::ceil(input.num_elements() as f32 / WORKGROUP_SIZE_DOUBLED as f32) as usize;
        let last_pass = reduced_size == 1;

        let reduced_sum = TypedBuffer::<u32>::new_zeroed(
            "reduced sum",
            reduced_size,
            BufferUsages::STORAGE,
            device,
        );

        let scan_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Label::from("scan bind group"),
            layout: scan_pipeline.bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: reduced_sum.buffer().as_entire_binding(),
                },
            ],
        });

        passes.push(StaticComputeEncodeDescriptor::new_1d(
            scan_pipeline.pipeline(),
            vec![scan_bind_group],
            reduced_size as u32,
        ));

        if !last_pass {
            Scan::create_recursive_bind_groups(
                &reduced_sum,
                scan_pipeline,
                sum_pipeline,
                passes,
                device,
            );

            let sum_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Label::from("sum bind group"),
                layout: sum_pipeline.bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: reduced_sum.buffer().as_entire_binding(),
                    },
                ],
            });

            passes.push(StaticComputeEncodeDescriptor::new_1d(
                sum_pipeline.pipeline(),
                vec![sum_bind_group],
                (input.num_elements() as f32 / WORKGROUP_SIZE as f32).ceil() as u32,
            ));
        }
    }

    pub fn encode(&self, command_encoder: &mut CommandEncoder) {
        let mut compute_pass: ComputePass =
            command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Label::from("Scan"),
            });
        self.encode_to_pass(&mut compute_pass);
    }

    pub fn encode_to_pass<'a, 'b>(&'a self, compute_pass: &mut ComputePass<'b>)
    where
        'a: 'b,
    {
        for p in &self.passes {
            p.encode(compute_pass);
        }
    }
}

// todo: remove (debug)!
pub async fn test_scan(ctx: &Arc<Gpu>) {
    let num_chunks = (1024 * 1024 * 1024) / (32 * 32 * 32);
    let data: Vec<u32> = (0..num_chunks)
        .map(|_| if js_sys::Math::random() > 0.5 { 1 } else { 0 })
        .collect();
    let mut scan_result = Vec::new();
    let mut prefix = 0;
    for d in &data {
        scan_result.push(prefix);
        prefix += d;
    }

    let buffer = TypedBuffer::from_data(
        "data",
        &data,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        ctx.device(),
    );
    let read_buffer = MappableBuffer::from_typed_buffer(&buffer, ctx.device());

    let prep = WGSLPreprocessor::default();
    let scan = Scan::new(&buffer, &prep, ctx);

    let mut command_encoder = ctx
        .device()
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Label::from("scan test"),
        });

    scan.encode(&mut command_encoder);

    command_encoder.copy_buffer_to_buffer(
        buffer.buffer(),
        0,
        read_buffer.buffer(),
        0,
        buffer.size(),
    );

    ctx.queue().submit(vec![command_encoder.finish()]);

    map_buffer(read_buffer.buffer(), ..).await;
    let mapped_range = read_buffer.buffer().slice(..).get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&mapped_range).to_vec();
    drop(mapped_range);

    let mut correct = true;
    if result.len() != scan_result.len() {
        log::error!("scan result not the same length!");
        correct = false;
    } else {
        for i in 0..scan_result.len() {
            if result[i] != scan_result[i] {
                log::error!(
                    "no the same at {}, cpu={}, gpu={}",
                    i,
                    scan_result[i],
                    result[i]
                );
                correct = false;
            }
        }
    }
    if correct {
        log::info!("SCAN CORRECT!!! HURRRAYY!!");
    }
}
