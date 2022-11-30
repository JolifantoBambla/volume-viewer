use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use glam::{UVec2, UVec3};
use wasm_bindgen::prelude::wasm_bindgen;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, Label, ShaderModuleDescriptor, ShaderSource};
use wgsl_preprocessor::WGSLPreprocessor;
use crate::renderer::context::GPUContext;
use crate::renderer::pass::{ComputePipelineData, GPUPass};
use crate::resource::buffer::{map_buffer, TypedBuffer};
use crate::resource::MappableBuffer;

const WORKGROUP_SIZE: u32 = 256;
const WORKGROUP_SIZE_DOUBLED: u32 = WORKGROUP_SIZE * 2;

pub struct Scan {
    ctx: Arc<GPUContext>,
    passes: Vec<ComputePipelineData>,
}

impl Scan {
    pub fn new(
        input_buffer: &TypedBuffer<u32>,
        wgsl_preprocessor: &WGSLPreprocessor,
        ctx: &Arc<GPUContext>,
    ) -> Self {
        if !input_buffer.supports(BufferUsages::STORAGE | BufferUsages::COPY_SRC) {
            panic!("buffer to scan needs to support STORAGE and COPY_SRC");
        }

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
        let scan_pipeline = Rc::new(ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Label::from("Scan"),
                layout: None,
                module: &scan_shader_module,
                entry_point: "main",
            }));
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
        let sum_pipeline = Rc::new(ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Label::from("Sum"),
                layout: None,
                module: &sum_shader_module,
                entry_point: "main",
            }));
        let sum_bind_group_layout = sum_pipeline.get_bind_group_layout(0);

        let output_max = TypedBuffer::new_single_element(
            "scan output",
            0,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            &ctx.device
        );

        let mut passes = Vec::new();
        Scan::create_recursive_bind_groups(
            &scan_pipeline,
            &sum_pipeline,
            &input_buffer,
            input_buffer.num_elements(),
            None,
            &output_max,
            &scan_bind_group_layout,
            &sum_bind_group_layout,
            &mut passes,
            &ctx.device,
        );

        Self {
            ctx: ctx.clone(),
            passes,
        }
    }

    fn create_scan_bind_group(
        scan_layout: &BindGroupLayout,
        input: &TypedBuffer<u32>,
        input_max: &TypedBuffer<u32>,
        reduced_sum: &TypedBuffer<u32>,
        reduced_max: &TypedBuffer<u32>,
        use_max_input: &TypedBuffer<u32>,
        device: &Device
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Label::from("scan bind group"),
            layout: scan_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_max.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: reduced_sum.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: reduced_max.buffer().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: use_max_input.buffer().as_entire_binding(),
                },
            ]
        })
    }

    fn create_recursive_bind_groups(
        scan_pipeline: &Rc<ComputePipeline>,
        sum_pipeline: &Rc<ComputePipeline>,
        input: &TypedBuffer<u32>,
        input_size: usize,
        input_max: Option<&TypedBuffer<u32>>,
        output_max: &TypedBuffer<u32>,
        scan_layout: &BindGroupLayout,
        sum_layout: &BindGroupLayout,
        passes: &mut Vec<ComputePipelineData>,
        device: &Device
    ) {
        let reduced_size = f32::ceil(input_size as f32 / WORKGROUP_SIZE_DOUBLED as f32) as usize;
        let last_pass = reduced_size == 1;

        let use_max_input: TypedBuffer<u32> = TypedBuffer::new_single_element(
            "use max input",
            if input_max.is_some() { 1 } else { 0 },
            BufferUsages::UNIFORM,
            device,
        );

        let reduced_sum = TypedBuffer::<u32>::new_zeroed(
            "reduced sum",
            reduced_size,
            BufferUsages::STORAGE,
            device,
        );

        let reduced_max = if last_pass {
            None
        } else {
            Some(TypedBuffer::new_zeroed(
                "reduced max",
                reduced_size,
                BufferUsages::STORAGE,
                device,
            ))
        };

        let in_max: Option<TypedBuffer<u32>> = if input_max.is_some() {
            None
        } else {
            Some(TypedBuffer::new_single_element(
                "input max",
                0,
                BufferUsages::STORAGE,
                device,
            ))
        };

        let scan_bind_group = Scan::create_scan_bind_group(
            scan_layout,
            input,
            if let Some(input_max) = input_max {
                input_max
            } else {
                in_max.as_ref().unwrap()
            },
            &reduced_sum,
            if last_pass {
                &output_max
            } else {
                reduced_max.as_ref().unwrap()
            },
            &use_max_input,
            device,
        );

        passes.push(ComputePipelineData::new_1d(
            scan_pipeline,
            vec![scan_bind_group],
            reduced_size as u32,
        ));

        if let Some(reduced_max) = &reduced_max {
            Scan::create_recursive_bind_groups(
                scan_pipeline,
                sum_pipeline,
                &reduced_sum,
                reduced_size,
                Some(reduced_max),
                output_max,
                scan_layout,
                sum_layout,
                passes,
                device,
            );
            
            let sum_bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Label::from("sum bind group"),
                layout: sum_layout,
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

            passes.push(ComputePipelineData::new_1d(
                sum_pipeline,
                vec![sum_bind_group],
                (input_size as f32 / WORKGROUP_SIZE as f32).ceil() as u32,
            ));
        }
    }

    pub fn encode(&self, command_encoder: &mut wgpu::CommandEncoder) {
        let mut cpass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from("Scan")
        });
        for p in &self.passes {
            p.encode(&mut cpass);
        }
    }
}


pub async fn test_scan(ctx: &Arc<GPUContext>) {
    let num_chunks = (1024*1024*1024) / (32*32*32);
    let data: Vec<u32> = (0..num_chunks).map(|_| if js_sys::Math::random() > 0.5 { 1 } else { 0 }).collect();
    let mut scan_result = Vec::new();
    let mut prefix = 0;
    for i in 0..data.len() {
        scan_result.push(prefix);
        prefix += data[i];
    }

    let buffer = TypedBuffer::from_data("data", &data, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST, &ctx.device);
    let read_buffer = buffer.create_read_buffer(&ctx.device);

    let prep = WGSLPreprocessor::default();
    let scan = Scan::new(&buffer, &prep, ctx);

    let mut command_encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Label::from("scan test")
    });

    scan.encode(&mut command_encoder);

    command_encoder.copy_buffer_to_buffer(
        buffer.buffer(),
        0,
        read_buffer.buffer(),
        0,
        buffer.size()
    );

    ctx.queue.submit(vec![command_encoder.finish()]);

    map_buffer(read_buffer.buffer(), ..)
        .await;
    let mapped_range = read_buffer.buffer().slice(..).get_mapped_range();
    let result : Vec<u32> = bytemuck::cast_slice(&mapped_range).to_vec();
    drop(mapped_range);

    let mut correct = true;
    if result.len() != scan_result.len() {
        log::error!("scan result not the same length!");
        correct = false;
    } else {
        for i in 0..scan_result.len() {
            if result[i] != scan_result[i] {
                log::error!("no the same at {}, cpu={}, gpu={}", i, scan_result[i], result[i]);
                correct = false;
            }
        }
    }
    if correct {
        log::info!("SCAN CORRECT!!! HURRRAYY!!");
    }
}
