use std::marker::PhantomData;
use std::sync::Arc;
use wgpu::{Buffer, BufferAddress, BufferUsages, Maintain, MaintainBase, MapMode};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use crate::renderer::context::GPUContext;

pub mod process_requests;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuListMeta {
    capacity: u32,
    fill_pointer: u32,
}

pub struct GpuList<T: bytemuck::Pod + bytemuck::Zeroable> {
    ctx: Arc<GPUContext>,
    capacity: u32,
    list_buffer: Buffer,
    meta_buffer: Buffer,
    phantom_data: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuList<T> {
    pub fn new(name: Option<&str>, capacity: u32, ctx: &Arc<GPUContext>) -> Self {
        let zeros = bytemuck::zeroed_vec::<T>(capacity as usize);
        let meta = vec![GpuListMeta {
            capacity,
            fill_pointer: 0,
        }];
        let list_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &bytemuck::cast_slice(zeros.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let meta_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &bytemuck::cast_slice(meta.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        Self {
            ctx: ctx.clone(),
            capacity,
            list_buffer,
            meta_buffer,
            phantom_data: PhantomData
        }
    }

    pub fn map_for_reading(&self) {
        self.list_buffer
            .slice(..)
            .map_async(MapMode::Read, || {});
        self.meta_buffer
            .slice(..)
            .map_async(MapMode::Read, || {});
    }

    /// read and unmaps buffer
    /// panics if `GpuList::map_for_reading` has not been called and device has not been polled until
    /// mapping finished
    pub fn read_mapped(&self) -> Vec<T> {
        let meta_view = self.meta_buffer
            .slice(..)
            .get_mapped_range();
        let list_view = self.list_buffer
            .slice(..)
            .get_mapped_range();

        let meta: Vec<GpuListMeta> = bytemuck::cast_slice(&meta_view).to_vec();
        let mut list: Vec<T> = bytemuck::cast_slice(&list_view).to_vec();

        drop(meta_view);
        drop(list_view);

        self.list_buffer.unmap();
        self.meta_buffer.unmap();

        list.shrink_to(meta[0].fill_pointer as usize);

        list
    }

    pub fn read(&self) -> Vec<T> {
        self.map_for_reading();
        self.ctx.device.poll(Maintain::Wait);
        self.read_mapped()
    }

    pub fn clear(&self) {
        let capacity = self.capacity;
        let zeros = bytemuck::zeroed_vec::<T>(capacity as usize);
        let meta = vec![GpuListMeta {
            capacity,
            fill_pointer: 0,
        }];
        self.ctx.queue.write_buffer(
            &self.meta_buffer,
            0 as BufferAddress,
            bytemuck::cast_slice(meta.as_slice())
        );
        self.ctx.queue.write_buffer(
            &self.list_buffer,
            0 as BufferAddress,
            bytemuck::cast_slice(zeros.as_slice())
        );
    }
}

