pub mod dvr;
pub mod present_to_screen;
pub mod scan;

use glam::{UVec2, UVec3};
use std::rc::Rc;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, ComputePass, ComputePipeline,
    ComputePipelineDescriptor, Device,
};
use wgpu_framework::context::Gpu;

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

#[derive(Debug)]
pub struct ComputeEncodeDescriptor {
    pipeline: Rc<ComputePipeline>,
    bind_groups: Vec<BindGroup>,
    work_group_size: UVec3,
}

impl ComputeEncodeDescriptor {
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
