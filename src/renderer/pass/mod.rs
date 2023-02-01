pub mod dvr;
pub mod present_to_screen;
pub mod scan;

use std::mem::size_of;
use glam::{UVec2, UVec3};
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, BufferAddress, BufferUsages, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::Buffer;

pub trait AsBindGroupEntries {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry>;
}

pub trait GPUPass<T: AsBindGroupEntries> {
    // todo: maybe move out of pub trait and use in other resources as well
    fn ctx(&self) -> &Arc<Gpu>;
    fn label(&self) -> &str;

    fn bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    fn create_bind_group(&self, resources: T) -> wgpu::BindGroup {
        self.ctx()
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.bind_group_layout(),
                entries: &resources.as_bind_group_entries(),
            })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DispatchWorkgroupsIndirect {
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
}

impl DispatchWorkgroupsIndirect {
    pub fn new_1d() -> Self {
        Self {
            workgroup_count_y: 1,
            workgroup_count_z: 1,
            ..Default::default()
        }
    }
    pub fn new_2d() -> Self {
        Self {
            workgroup_count_z: 1,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
pub struct ComputeEncodeIndirectDescriptor {
    pipeline: Rc<ComputePipeline>,
    bind_groups: Vec<BindGroup>,
    indirect_buffer: Rc<Buffer<DispatchWorkgroupsIndirect>>,
    indirect_offset: BufferAddress,
}

impl ComputeEncodeIndirectDescriptor {
    pub fn new(pipeline: &Rc<ComputePipeline>, bind_groups: Vec<BindGroup>, gpu: &Arc<Gpu>) -> Self {
        let indirect_buffer = Rc::new(Buffer::new_zeroed(
            "indirect buffer",
            1,
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
            gpu,
        ));
        Self::with_indirect_buffer(pipeline, bind_groups, &indirect_buffer, 0)
    }
    pub fn with_indirect_buffer(pipeline: &Rc<ComputePipeline>, bind_groups: Vec<BindGroup>, indirect_buffer: &Rc<Buffer<DispatchWorkgroupsIndirect>>, indirect_offset: BufferAddress) -> Self {
        assert!(indirect_buffer.supports(BufferUsages::INDIRECT));
        assert!(indirect_buffer.size() > indirect_offset);
        assert!((indirect_buffer.size() - indirect_offset) as usize >= size_of::<u32>() * 3);
        Self { pipeline: pipeline.clone(), bind_groups, indirect_buffer: indirect_buffer.clone(), indirect_offset }
    }
    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        for (i, b) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, b, &[]);
        }
        compute_pass.dispatch_workgroups_indirect(
            self.indirect_buffer.buffer(),
            self.indirect_offset,
        );
    }
}


#[derive(Debug)]
pub struct StaticComputeEncodeDescriptor {
    pipeline: Rc<ComputePipeline>,
    bind_groups: Vec<BindGroup>,
    work_group_size: UVec3,
}

impl StaticComputeEncodeDescriptor {
    pub fn new(
        pipeline: &Rc<ComputePipeline>,
        bind_groups: Vec<BindGroup>,
        work_group_size: UVec3,
    ) -> Self {
        Self {
            pipeline: pipeline.clone(),
            bind_groups,
            work_group_size,
        }
    }

    pub fn new_1d(
        pipeline: &Rc<ComputePipeline>,
        bind_groups: Vec<BindGroup>,
        work_group_size: u32,
    ) -> Self {
        Self::new(pipeline, bind_groups, UVec3::new(work_group_size, 1, 1))
    }

    pub fn new_2d(
        pipeline: &Rc<ComputePipeline>,
        bind_groups: Vec<BindGroup>,
        work_group_size: UVec2,
    ) -> Self {
        Self::new(pipeline, bind_groups, work_group_size.extend(1))
    }

    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        for (i, b) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, b, &[]);
        }
        compute_pass.dispatch_workgroups(
            self.work_group_size.x,
            self.work_group_size.y,
            self.work_group_size.z,
        );
    }
}

#[derive(Debug)]
pub struct DynamicComputeEncodeDescriptor {
    pipeline: Rc<ComputePipeline>,
    bind_groups: Vec<BindGroup>,
}

impl DynamicComputeEncodeDescriptor {
    pub fn new(
        pipeline: &Rc<ComputePipeline>,
        bind_groups: Vec<BindGroup>,
    ) -> Self {
        Self {
            pipeline: pipeline.clone(),
            bind_groups,
        }
    }
    pub fn encode_1d<'a>(&'a self, compute_pass: &mut ComputePass<'a>, work_group_size: u32) {
        self.encode(compute_pass, UVec3::new(work_group_size, 1, 1));
    }
    pub fn encode_2d<'a>(&'a self, compute_pass: &mut ComputePass<'a>, work_group_size: UVec2) {
        self.encode(compute_pass, work_group_size.extend(1));
    }
    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>, work_group_size: UVec3) {
        compute_pass.set_pipeline(&self.pipeline);
        for (i, b) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, b, &[]);
        }
        compute_pass.dispatch_workgroups(
            work_group_size.x,
            work_group_size.y,
            work_group_size.z,
        );
    }
}

#[derive(Debug)]
pub enum ComputePassData {
    Direct(StaticComputeEncodeDescriptor),
    Indirect(ComputeEncodeIndirectDescriptor),
}

impl ComputePassData {
    pub fn encode<'a>(&'a self, compute_pass: &mut ComputePass<'a>) {
        match self {
            ComputePassData::Direct(d) => d.encode(compute_pass),
            ComputePassData::Indirect(i) => i.encode(compute_pass),
        }
    }
}

#[derive(Debug)]
pub struct ComputePipelineData<const N: usize> {
    pipeline: Rc<ComputePipeline>,
    bind_group_layouts: Vec<BindGroupLayout>,
}

impl<const N: usize> ComputePipelineData<N> {
    pub fn new(pipeline_descriptor: &ComputePipelineDescriptor, device: &Device) -> Self {
        let pipeline = Rc::new(device.create_compute_pipeline(pipeline_descriptor));
        let mut bind_group_layouts = Vec::new();
        for i in 0..N as u32 {
            bind_group_layouts.push(pipeline.get_bind_group_layout(i));
        }
        Self {
            pipeline,
            bind_group_layouts,
        }
    }

    pub fn pipeline(&self) -> &Rc<ComputePipeline> {
        &self.pipeline
    }

    pub fn bind_group_layout(&self, i: usize) -> &BindGroupLayout {
        assert!(i < N);
        &self.bind_group_layouts[i]
    }
}
