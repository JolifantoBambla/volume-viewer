use crate::renderer::pass::AsBindGroupEntries;
use crate::resource::buffer::TypedBuffer;
use crate::resource::MappableBuffer;
use std::cmp::min;
use std::sync::Arc;
use wgpu::{
    BindGroupEntry, BindingResource, BufferAddress, BufferUsages, CommandEncoder, Maintain, MapMode,
};
use wgpu_framework::context::Gpu;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuListMeta {
    capacity: u32,
    fill_pointer: u32,
    written_at: u32,
}

#[derive(Debug)]
pub struct GpuListReadResult<T: bytemuck::Pod + bytemuck::Zeroable> {
    list: Vec<T>,
    written_at: u32,
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> From<GpuListReadResult<T>> for (Vec<T>, u32) {
    fn from(read_result: GpuListReadResult<T>) -> Self {
        (read_result.list, read_result.written_at)
    }
}

#[derive(Debug)]
pub struct GpuList<T: bytemuck::Pod + bytemuck::Zeroable> {
    ctx: Arc<Gpu>,
    capacity: u32,
    list_buffer: TypedBuffer<T>,
    list_read_buffer: MappableBuffer<T>,
    meta_buffer: TypedBuffer<GpuListMeta>,
    meta_read_buffer: MappableBuffer<GpuListMeta>,
}

impl<T: bytemuck::Pod> GpuList<T> {
    pub fn new(label: &str, capacity: u32, ctx: &Arc<Gpu>) -> Self {
        let meta = GpuListMeta {
            capacity,
            fill_pointer: 0,
            written_at: 0,
        };
        let list_buffer = TypedBuffer::new_zeroed(
            format!("{label} [list]").as_str(),
            capacity as usize,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ctx.device(),
        );
        let meta_buffer = TypedBuffer::new_single_element(
            format!("{label} [meta]").as_str(),
            meta,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ctx.device(),
        );

        let list_read_buffer = MappableBuffer::from_typed_buffer(&list_buffer, ctx.device());
        let meta_read_buffer = MappableBuffer::from_typed_buffer(&meta_buffer, ctx.device());

        Self {
            ctx: ctx.clone(),
            capacity,
            list_buffer,
            list_read_buffer,
            meta_buffer,
            meta_read_buffer,
        }
    }

    pub fn copy_to_readable(&self, encoder: &mut CommandEncoder) {
        if self.list_read_buffer.is_ready() && self.meta_read_buffer.is_ready() {
            encoder.copy_buffer_to_buffer(
                self.meta_buffer.buffer(),
                0,
                self.meta_read_buffer.buffer(),
                0,
                self.meta_buffer.size(),
            );
            encoder.copy_buffer_to_buffer(
                self.list_buffer.buffer(),
                0,
                self.list_read_buffer.buffer(),
                0,
                self.list_buffer.size(),
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
    pub fn read_mapped(&self) -> Option<GpuListReadResult<T>> {
        if self.list_read_buffer.is_mapped() && self.meta_read_buffer.is_mapped() {
            let meta = self.meta_read_buffer.maybe_read_all()[0];
            let mut list = self.list_read_buffer.maybe_read_all();

            list.truncate(min(meta.fill_pointer, self.capacity) as usize);

            Some(GpuListReadResult {
                list,
                written_at: meta.written_at,
            })
        } else {
            None
        }
    }

    pub fn read(&self) -> Option<GpuListReadResult<T>> {
        self.map_for_reading();
        self.ctx.device().poll(Maintain::Wait);
        self.read_mapped()
    }

    pub fn clear(&self) {
        let capacity = self.capacity;
        let meta = GpuListMeta {
            capacity,
            fill_pointer: 0,
            written_at: 0,
        };
        self.ctx.queue().write_buffer(
            self.meta_buffer.buffer(),
            0 as BufferAddress,
            bytemuck::bytes_of(&meta),
        );
    }

    pub fn meta_as_binding(&self) -> BindingResource {
        self.meta_buffer.buffer().as_entire_binding()
    }

    pub fn list_as_binding(&self) -> BindingResource {
        self.list_buffer.buffer().as_entire_binding()
    }
}

// todo: use wgpu_framework::Buffer instead which implements Drop
impl<T: bytemuck::Pod + bytemuck::Zeroable> Drop for GpuList<T> {
    fn drop(&mut self) {
        self.list_buffer.buffer().destroy();
        self.list_read_buffer.buffer().destroy();
        self.meta_buffer.buffer().destroy();
        self.meta_read_buffer.buffer().destroy();
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
