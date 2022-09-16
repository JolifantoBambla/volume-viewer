use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferAddress, MapMode};


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
