use std::marker::PhantomData;
use std::mem;
use std::mem::size_of;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, Device, Label, MapMode};

pub struct TypedBuffer<T: bytemuck::Pod> {
    label: String,
    buffer: Buffer,
    num_elements: usize,
    phantom_data: PhantomData<T>
}

impl<T: bytemuck::Pod> TypedBuffer<T> {
    pub fn new_zeroed(label: &str, num_elements: usize, usage: BufferUsages, device: &Device) -> Self {
        let data = vec![unsafe { mem::zeroed() }; num_elements];
        TypedBuffer::from_data(label, &data, usage, device)
    }

    pub fn new_single_element(label: &str, element: T, usage: BufferUsages, device: &Device) -> Self {
        let data = vec![element];
        TypedBuffer::from_data(label, &data, usage, device)
    }

    pub fn from_data(label: &str, data: &Vec<T>, usage: BufferUsages, device: &Device) -> Self {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Label::from(label),
            contents: bytemuck::cast_slice(data),
            usage
        });
        Self {
            label: String::from(label),
            buffer,
            num_elements: data.len(),
            phantom_data: PhantomData,
        }
    }

    fn create_read_buffer(&self, device: &Device) -> Self {
        assert!(self.supports(BufferUsages::COPY_SRC));
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Label::from("map buffer"),
            size: self.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        Self {
            label: format!("Map buffer [{}]", self.label.as_str()),
            buffer,
            num_elements: self.num_elements,
            phantom_data: PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buffer {
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

pub async fn map_buffer<S: RangeBounds<BufferAddress>>(buffer: &Buffer, bounds: S) {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    buffer.slice(bounds).map_async(MapMode::Read, move |v| {
        sender.send(v).expect("Could not report buffer mapping");
    });
    receiver
        .receive()
        .await
        .expect("Could not map buffer")
        .expect("I said: could not map buffer!");
}

#[derive(Eq, PartialEq)]
pub enum BufferState {
    Ready,
    Mapping,
    Mapped,
}

pub struct MappableBufferState {
    state: BufferState,
}

pub struct MappableBuffer<T: bytemuck::Pod> {
    buffer: TypedBuffer<T>,
    state: Arc<Mutex<MappableBufferState>>,
}

impl<T: bytemuck::Pod> MappableBuffer<T> {
    pub fn from_buffer(buffer: &TypedBuffer<T>, device: &Device) -> Self {
        Self {
            buffer: buffer.create_read_buffer(device),
            state: Arc::new(Mutex::new(MappableBufferState {
                state: BufferState::Ready,
            })),
        }
    }

    pub fn buffer(&self) -> &Buffer {
        self.buffer.buffer()
    }

    pub fn size(&self) -> BufferAddress {
        self.buffer.size()
    }

    /// Maps a mappable buffer if it is not already mapped or being mapped.
    /// The buffer can be read
    pub fn map_async<S: RangeBounds<BufferAddress>>(&self, mode: MapMode, bounds: S) {
        let s = self.state.clone();
        if self.is_ready() {
            s.lock().unwrap().state = BufferState::Mapping;
            self.buffer().slice(bounds).map_async(mode, move |_| {
                s.lock().unwrap().state = BufferState::Mapped;
            });
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Ready
    }

    pub fn is_mapped(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Mapped
    }

    pub fn maybe_read<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Vec<T> {
        if self.is_mapped() {
            let mapped_range = self.buffer().slice(bounds).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            result
        } else {
            Vec::new()
        }
    }

    pub fn unmap(&self) {
        if self.is_mapped() {
            self.buffer().unmap();
            self.state.lock().unwrap().state = BufferState::Ready;
        }
    }

    pub fn maybe_read_all(&self) -> Vec<T> {
        let result = self.maybe_read(..);
        self.unmap();
        result
    }
}

pub struct MultiBufferedMappableBuffer<T: bytemuck::Pod> {
    buffers: Vec<MappableBuffer<T>>,
    buffer_size: BufferAddress,
}

impl<T: bytemuck::Pod> MultiBufferedMappableBuffer<T> {
    pub fn from_buffer(buffer: &TypedBuffer<T>, num_buffers: u32, device: &Device) -> Self {
        assert!(num_buffers > 0);
        let buffers = (0..num_buffers)
            .map(|_| MappableBuffer::from_buffer(buffer, device))
            .collect();
        Self {
            buffers,
            buffer_size: buffer.size(),
        }
    }

    pub fn to_index(&self, index_or_frame_number: u32) -> usize {
        index_or_frame_number as usize % self.num_buffers()
    }

    fn get_buffer(&self, index_or_frame_number: u32) -> &MappableBuffer<T> {
        &self.buffers[self.to_index(index_or_frame_number)]
    }

    pub fn to_previous_index(&self, index_or_frame_number: u32) -> u32 {
        let num_buffers = self.num_buffers() as u32;
        (index_or_frame_number + num_buffers - 1) % num_buffers
    }

    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    pub fn as_buffer_ref(&self, index_or_frame_number: u32) -> &Buffer {
        self.get_buffer(index_or_frame_number).buffer()
    }

    pub fn map_async<S: RangeBounds<BufferAddress>>(
        &self,
        index_or_frame_number: u32,
        mode: MapMode,
        bounds: S,
    ) {
        self.get_buffer(index_or_frame_number)
            .map_async(mode, bounds);
    }

    pub fn is_ready(&self, index_or_frame_number: u32) -> bool {
        self.get_buffer(index_or_frame_number).is_ready()
    }

    pub fn is_mapped(&self, index_or_frame_number: u32) -> bool {
        self.get_buffer(index_or_frame_number).is_mapped()
    }

    pub fn maybe_read<S: RangeBounds<BufferAddress>>(
        &self,
        index_or_frame_number: u32,
        bounds: S,
    ) -> Vec<T> {
        self.get_buffer(index_or_frame_number).maybe_read(bounds)
    }

    pub fn unmap(&self, index_or_frame_number: u32) {
        self.get_buffer(index_or_frame_number).unmap();
    }

    pub fn maybe_read_all(&self, index_or_frame_number: u32) -> Vec<T> {
        self.get_buffer(index_or_frame_number).maybe_read_all()
    }
}
