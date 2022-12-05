use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::mem::size_of;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder, Device, Label, MapMode};

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
        let label = format!("map buffer [{}]", self.label.as_str());
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Label::from(label.as_str()),
            size: self.size(),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        Self {
            label,
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

#[derive(Debug, Eq, PartialEq)]
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

#[derive(Debug, Eq, PartialEq)]
pub enum BufferMapError {
    NotReady,
    NotMapped,
}

pub struct ReadableStorageBuffer<T: bytemuck::Pod> {
    storage_buffer: Arc<TypedBuffer<T>>,
    read_buffer: MappableBuffer<T>,
}

impl<T: bytemuck::Pod> ReadableStorageBuffer<T> {
    pub fn new(label: &str, capacity: usize, device: &Device) -> Self {
        let storage_buffer = Arc::new(TypedBuffer::new_zeroed(
            label,
            capacity,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            device,
        ));
        let read_buffer = MappableBuffer::from_buffer(&storage_buffer, device);
        Self {
            storage_buffer,
            read_buffer,
        }
    }

    pub fn from_data(label: &str, data: &Vec<T>, device: &Device) -> Self {
        let storage_buffer = Arc::new(TypedBuffer::from_data(
            label,
            data,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            device,
        ));
        let read_buffer = MappableBuffer::from_buffer(&storage_buffer, device);
        Self {
            storage_buffer,
            read_buffer,
        }
    }

    pub fn storage_buffer(&self) -> &Arc<TypedBuffer<T>> {
        &self.storage_buffer
    }

    pub fn read_buffer(&self) -> &MappableBuffer<T> {
        &self.read_buffer
    }

    pub fn buffer(&self) -> &Buffer {
        self.storage_buffer.buffer()
    }

    pub fn size(&self) -> BufferAddress {
        self.storage_buffer.size()
    }

    pub fn copy_to_readable(&self, command_encoder: &mut CommandEncoder) -> Result<(), BufferMapError> {
        if self.read_buffer.is_ready() {
            command_encoder.copy_buffer_to_buffer(
                self.storage_buffer.buffer(),
                0,
                self.read_buffer.buffer(),
                0,
                self.storage_buffer.size(),
            );
            Ok(())
        } else {
            Err(BufferMapError::NotReady)
        }
    }

    /// Maps a mappable buffer if it is not already mapped or being mapped.
    /// The buffer can be read
    pub fn map_async<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Result<(), BufferMapError> {
        if self.read_buffer.is_ready() {
            let s = self.read_buffer.state.clone();
            s.lock().unwrap().state = BufferState::Mapping;
            self.read_buffer.buffer().slice(bounds).map_async(MapMode::Read, move |_| {
                s.lock().unwrap().state = BufferState::Mapped;
            });
            Ok(())
        }  else {
            Err(BufferMapError::NotReady)
        }
    }

    pub fn map_all_async(&self) -> Result<(), BufferMapError> {
        self.map_async(..)
    }

    pub fn read<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Result<Vec<T>, BufferMapError> {
        if self.read_buffer.is_mapped() {
            let mapped_range = self.read_buffer.buffer().slice(bounds).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            Ok(result)
        } else {
            Err(BufferMapError::NotMapped)
        }
    }

    #[inline]
    fn unmap_unchecked(&self) {
        self.read_buffer.buffer().unmap();
        self.read_buffer.state.lock().unwrap().state = BufferState::Ready;
    }

    pub fn unmap(&self) {
        if self.read_buffer.is_mapped() {
            self.unmap_unchecked()
        }
    }

    pub fn read_all(&self) -> Result<Vec<T>, BufferMapError> {
        let result = self.read(..);
        if result.is_ok() {
            self.unmap_unchecked()
        }
        result
    }
}

impl<T: bytemuck::Pod> Display for ReadableStorageBuffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Readable Storage Buffer [{}] (size: {}, state: {:?})",
               self.storage_buffer.label,
               self.size(),
               self.read_buffer.state.lock().unwrap().state
        )
    }
}
