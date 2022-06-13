use std::borrow::Cow;
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, CommandEncoder, Label, TextureView};
use crate::renderer::context::GPUContext;

pub trait AsBindGroupEntries {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry>;
}

pub trait GPUPass<T: AsBindGroupEntries> {
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


