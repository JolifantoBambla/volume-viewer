pub mod dvr;
pub mod present_to_screen;
pub mod ray_guided_dvr;
pub mod scan;

use std::rc::Rc;
use crate::renderer::context::GPUContext;
use std::sync::Arc;
use glam::{UVec2, UVec3};
use wgpu::{BindGroup, BindGroupEntry, ComputePipeline};

pub trait AsBindGroupEntries {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry>;
}

pub trait GPUPass<T: AsBindGroupEntries> {
    // todo: maybe move out of pub trait and use in other resources as well
    fn ctx(&self) -> &Arc<GPUContext>;
    fn label(&self) -> &str;

    fn bind_group_layout(&self) -> &wgpu::BindGroupLayout;
    fn create_bind_group(&self, resources: T) -> wgpu::BindGroup {
        self.ctx()
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.bind_group_layout(),
                entries: &resources.as_bind_group_entries(),
            })
    }
}

pub struct ComputePipelineData {
    pipeline: Rc<ComputePipeline>,
    bind_groups: Vec<BindGroup>,
    work_group_size: UVec3,
}

impl ComputePipelineData {
    pub fn new(pipeline: &Rc<ComputePipeline>, bind_groups: Vec<BindGroup>, work_group_size: UVec3) -> Self {
        Self {
            pipeline: pipeline.clone(),
            bind_groups,
            work_group_size,
        }
    }

    pub fn new_1d(pipeline: &Rc<ComputePipeline>, bind_groups: Vec<BindGroup>, work_group_size: u32) -> Self {
        Self::new(pipeline, bind_groups, UVec3::new(work_group_size, 1, 1))
    }

    pub fn new_2d(pipeline: &Rc<ComputePipeline>, bind_groups: Vec<BindGroup>, work_group_size: UVec2) -> Self {
        Self::new(pipeline, bind_groups, work_group_size.extend(1))
    }

    pub fn encode<'a>(&'a self, compute_pass: &mut wgpu::ComputePass<'a>) {
        compute_pass.set_pipeline(&self.pipeline);
        for (i, b) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(i as u32, b, &[]);
        }
        compute_pass.dispatch_workgroups(
            self.work_group_size.x,
            self.work_group_size.y,
            self.work_group_size.z
        );
    }
}
