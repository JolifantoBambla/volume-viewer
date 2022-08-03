use crate::renderer::context::GPUContext;
use crate::renderer::pass::AsBindGroupEntries;
use futures::future::join_all;
use std::cmp::min;
use std::marker::PhantomData;
use std::mem::size_of;
use std::process::Command;
use std::sync::Arc;
use js_sys::Map;
use wasm_bindgen_futures::spawn_local;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoder, Maintain, MaintainBase, MapMode,
};
use crate::renderer::resources::{map_buffer, MappableBuffer};

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
    list_read_buffer: MappableBuffer<T>,
    list_buffer_size: BufferAddress,
    meta_buffer: Buffer,
    meta_read_buffer: MappableBuffer<GpuListMeta>,
    phantom_data: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuList<T> {
    pub fn new(name: Option<&str>, capacity: u32, ctx: &Arc<GPUContext>) -> Self {
        let list_buffer_size = (size_of::<T>() * capacity as usize) as BufferAddress;
        let meta = GpuListMeta {
            capacity,
            fill_pointer: 0,
        };
        let list_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: list_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let meta_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &bytemuck::bytes_of(&meta),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let list_read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: list_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let meta_read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<GpuListMeta>() as BufferAddress,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx: ctx.clone(),
            capacity,
            list_buffer,
            list_read_buffer: MappableBuffer::new(list_read_buffer),
            list_buffer_size,
            meta_buffer,
            meta_read_buffer: MappableBuffer::new(meta_read_buffer),
            phantom_data: PhantomData,
        }
    }

    pub fn copy_to_readable(&self, encoder: &mut CommandEncoder) {
        if self.list_read_buffer.is_ready() && self.meta_read_buffer.is_ready() {
            encoder.copy_buffer_to_buffer(
                &self.meta_buffer,
                0,
                self.meta_read_buffer.as_buffer_ref(),
                0,
                size_of::<GpuListMeta>() as BufferAddress,
            );
            encoder.copy_buffer_to_buffer(
                &self.list_buffer,
                0,
                self.list_read_buffer.as_buffer_ref(),
                0,
                self.list_buffer_size,
            );
        }
    }

    pub fn map_for_reading(&self) {
        if self.list_read_buffer.is_ready() && self.meta_read_buffer.is_ready() {
            self.list_read_buffer.map_async(MapMode::Read, ..);
            self.meta_read_buffer.map_async(MapMode::Read, ..);
        }
    }

    /// read and unmaps buffer
    /// panics if `GpuList::map_for_reading` has not been called and device has not been polled until
    /// mapping finished
    pub fn read_mapped(&self) -> Vec<T> {
        if self.list_read_buffer.is_mapped() && self.meta_read_buffer.is_mapped() {
            let meta = self.meta_read_buffer.maybe_read_all()[0];
            let mut list = self.list_read_buffer.maybe_read_all();

            list.truncate(min(meta.fill_pointer, self.capacity) as usize);

            list
        } else {
            Vec::new()
        }
    }

    pub fn read(&self) -> Vec<T> {
        self.map_for_reading();
        self.ctx.device.poll(Maintain::Wait);
        self.read_mapped()
    }

    pub fn clear(&self) {
        let capacity = self.capacity;
        let meta = GpuListMeta {
            capacity,
            fill_pointer: 0,
        };
        self.ctx.queue.write_buffer(
            &self.meta_buffer,
            0 as BufferAddress,
            bytemuck::bytes_of(&meta),
        );
    }

    pub fn meta_as_binding(&self) -> BindingResource {
        self.meta_buffer.as_entire_binding()
    }

    pub fn list_as_binding(&self) -> BindingResource {
        self.list_buffer.as_entire_binding()
    }
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> AsBindGroupEntries for GpuList<T> {
    fn as_bind_group_entries(&self) -> Vec<BindGroupEntry> {
        vec![
            BindGroupEntry {
                binding: 0,
                resource: self.list_as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: self.meta_as_binding(),
            },
        ]
    }
}
