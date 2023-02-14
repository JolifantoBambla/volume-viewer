use crate::context::Gpu;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::mem::size_of;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BufferAddress, BufferDescriptor, BufferUsages, Label, MapMode};

#[derive(Debug)]
pub struct Buffer<T: bytemuck::Pod> {
    gpu: Arc<Gpu>,
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
        Buffer::from_data(label, data.as_slice(), usage, ctx)
    }

    pub fn new_single_element(
        label: &str,
        element: T,
        usage: BufferUsages,
        ctx: &Arc<Gpu>,
    ) -> Self {
        let data = vec![element];
        Buffer::from_data(label, data.as_slice(), usage, ctx)
    }

    pub fn from_data(label: &str, data: &[T], usage: BufferUsages, ctx: &Arc<Gpu>) -> Self {
        let buffer = ctx.device().create_buffer_init(&BufferInitDescriptor {
            label: Label::from(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self {
            gpu: ctx.clone(),
            label: String::from(label),
            buffer,
            num_elements: data.len(),
            phantom_data: PhantomData,
        }
    }

    pub fn create_read_buffer(&self) -> Self {
        assert!(self.supports(BufferUsages::COPY_SRC));
        let label = format!("map buffer [{}]", self.label.as_str());
        let buffer = self.gpu.device().create_buffer(&BufferDescriptor {
            label: Label::from(label.as_str()),
            size: self.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            gpu: self.gpu.clone(),
            label,
            buffer,
            num_elements: self.num_elements,
            phantom_data: PhantomData,
        }
    }

    pub fn write_buffer(&self, data: &[T]) {
        self.write_buffer_with_offset(data, 0);
    }

    pub fn write_buffer_with_offset(&self, data: &[T], offset: BufferAddress) {
        self.gpu
            .queue()
            .write_buffer(self.buffer(), offset, bytemuck::cast_slice(data));
    }

    pub fn label(&self) -> &str {
        &self.label
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

#[derive(Debug, Eq, PartialEq)]
pub enum BufferMapError {
    NotReady,
    NotMapped,
}

impl Display for BufferMapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferMapError::NotReady => write!(f, "Buffer is not in 'Ready' state."),
            BufferMapError::NotMapped => write!(f, "Buffer is not in 'Mapped' state."),
        }
    }
}

impl Error for BufferMapError {}

#[derive(Debug, Eq, PartialEq)]
pub enum BufferState {
    /// The buffer is ready to be used in a command.
    Ready,

    /// The buffer is in the process of being mapped.
    Mapping,

    /// The buffer is mapped. It is safe to read its mapped range.
    Mapped,
}

#[derive(Debug)]
pub struct MappableBufferState {
    state: BufferState,
}

#[derive(Debug)]
pub struct MappableBuffer<T: bytemuck::Pod> {
    buffer: Buffer<T>,
    state: Arc<Mutex<MappableBufferState>>,
}

impl<T: bytemuck::Pod> MappableBuffer<T> {
    pub fn from_buffer(buffer: &Buffer<T>) -> Self {
        Self {
            buffer: buffer.create_read_buffer(),
            state: Arc::new(Mutex::new(MappableBufferState {
                state: BufferState::Ready,
            })),
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        self.buffer.buffer()
    }

    pub fn size(&self) -> BufferAddress {
        self.buffer.size()
    }

    /// Maps a mappable buffer if it is not already mapped or being mapped.
    /// The buffer can be read
    pub fn map_async<S: RangeBounds<BufferAddress>>(
        &self,
        mode: MapMode,
        bounds: S,
    ) -> Result<(), BufferMapError> {
        let s = self.state.clone();
        if self.is_ready() {
            s.lock().unwrap().state = BufferState::Mapping;
            self.buffer().slice(bounds).map_async(mode, move |_| {
                s.lock().unwrap().state = BufferState::Mapped;
            });
            Ok(())
        } else {
            Err(BufferMapError::NotReady)
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Ready
    }

    pub fn is_mapped(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Mapped
    }

    pub fn read<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Result<Vec<T>, BufferMapError> {
        if self.is_mapped() {
            let mapped_range = self.buffer().slice(bounds).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            Ok(result)
        } else {
            Err(BufferMapError::NotMapped)
        }
    }

    pub fn unmap(&self) {
        if self.is_mapped() {
            self.buffer().unmap();
            self.state.lock().unwrap().state = BufferState::Ready;
        }
    }

    pub fn read_all(&self) -> Result<Vec<T>, BufferMapError> {
        let result = self.read(..);
        if result.is_ok() {
            self.unmap();
        }
        result
    }
}
