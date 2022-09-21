use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferAddress, BufferUsages, Device, MapMode};

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
    buffer: Buffer,
    state: Arc<Mutex<MappableBufferState>>,
    phantom_data: PhantomData<T>,
}

impl<T: bytemuck::Pod> MappableBuffer<T> {
    pub fn new(buffer: Buffer) -> Self {
        Self {
            buffer,
            state: Arc::new(Mutex::new(MappableBufferState {
                state: BufferState::Ready,
            })),
            phantom_data: PhantomData,
        }
    }

    pub fn as_buffer_ref(&self) -> &Buffer {
        &self.buffer
    }

    /// Maps a mappable buffer if it is not already mapped or being mapped.
    /// The buffer can be read
    pub fn map_async<S: RangeBounds<BufferAddress>>(&self, mode: MapMode, bounds: S) {
        let s = self.state.clone();
        if self.is_ready() {
            s.lock().unwrap().state = BufferState::Mapping;
            self.buffer.slice(bounds).map_async(mode, move |_| {
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
            let mapped_range = self.buffer.slice(bounds).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            result
        } else {
            Vec::new()
        }
    }

    pub fn unmap(&self) {
        if self.is_mapped() {
            self.buffer.unmap();
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
    pub fn new(num_buffers: u32, init_data: &Vec<T>, device: &Device) -> Self {
        let num_entries = init_data.len();
        let buffer_size = (size_of::<T>() * num_entries as usize) as BufferAddress;

        let mut buffers: Vec<MappableBuffer<T>> = Vec::new();
        for _ in 0..num_buffers {
            let buffer = MappableBuffer::new(device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(init_data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            }));
            buffers.push(buffer);
        }

        Self {
            buffers,
            buffer_size,
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
        self.get_buffer(index_or_frame_number).as_buffer_ref()
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
