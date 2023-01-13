use crate::context::Gpu;
use std::marker::PhantomData;
use std::mem;
use std::mem::size_of;
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferAddress, BufferDescriptor, BufferUsages, Label};

#[derive(Debug)]
pub struct Buffer<T: bytemuck::Pod> {
    ctx: Arc<Gpu>,
    #[allow(unused)]
    label: String,
    buffer: wgpu::Buffer,
    num_elements: usize,
    phantom_data: PhantomData<T>,
}

impl<T: bytemuck::Pod> Buffer<T> {
    pub fn new_zeroed(
        label: &str,
        num_elements: usize,
        usage: BufferUsages,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let data = vec![unsafe { mem::zeroed() }; num_elements];
        Buffer::from_data(label, &data, usage, ctx)
    }

    pub fn new_single_element(
        label: &str,
        element: T,
        usage: BufferUsages,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let data = vec![element];
        Buffer::from_data(label, &data, usage, ctx)
    }

    pub fn from_data(label: &str, data: &Vec<T>, usage: BufferUsages, ctx: &Arc<Gpu>) -> Self {
        let buffer = ctx.device().create_buffer_init(&BufferInitDescriptor {
            label: Label::from(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self {
            ctx: ctx.clone(),
            label: String::from(label),
            buffer,
            num_elements: data.len(),
            phantom_data: PhantomData,
        }
    }

    #[allow(unused)]
    fn create_read_buffer(&self, ctx: &Arc<Gpu>) -> Self {
        assert!(self.supports(BufferUsages::COPY_SRC));
        let label = format!("map buffer [{}]", self.label.as_str());
        let buffer = ctx.device().create_buffer(&BufferDescriptor {
            label: Label::from(label.as_str()),
            size: self.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            ctx: ctx.clone(),
            label,
            buffer,
            num_elements: self.num_elements,
            phantom_data: PhantomData,
        }
    }

    pub fn write_buffer(&self, data: &Vec<T>) {
        self.write_buffer_with_offset(data.as_slice(), 0);
    }

    pub fn write_buffer_with_offset(&self, data: &[T], offset: BufferAddress) {
        self.ctx
            .queue()
            .write_buffer(self.buffer(), offset, bytemuck::cast_slice(data));
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn usage(&self) -> BufferUsages {
        self.buffer.usage()
    }

    pub fn size(&self) -> BufferAddress {
        self.buffer.size()
    }

    pub fn element_size(&self) -> usize {
        size_of::<T>()
    }

    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    pub fn supports(&self, usage: BufferUsages) -> bool {
        self.usage().contains(usage)
    }
}

impl<T: bytemuck::Pod> Drop for Buffer<T> {
    fn drop(&mut self) {
        self.buffer.destroy();
    }
}
