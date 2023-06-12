use crate::context::Gpu;
use crate::gpu::buffer::{Buffer, BufferMapError, MappableBuffer};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use wgpu::{BufferUsages, CommandEncoder, Label, QuerySet, QuerySetDescriptor, QueryType};

#[derive(Debug)]
pub struct CapacityReachedError {
    capacity: usize,
    index: usize,
}

impl Display for CapacityReachedError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimestampQuerySet has capacity {} but 'write_timestamp' was called {} times",
            self.capacity, self.index
        )
    }
}

impl Error for CapacityReachedError {}

pub struct TimeStampQuerySet {
    query_set: Arc<QuerySet>,
    resolve_buffer: Buffer<u64>,
    label: String,
    capacity: usize,
    next_index: usize,
}

impl TimeStampQuerySet {
    pub fn new(label: &str, capacity: usize, gpu: &Arc<Gpu>) -> Self {
        Self {
            query_set: Arc::new(gpu.device().create_query_set(&QuerySetDescriptor {
                label: Label::from(label),
                ty: QueryType::Timestamp,
                count: capacity as u32,
            })),
            resolve_buffer: Buffer::new_zeroed(
                label,
                capacity,
                BufferUsages::STORAGE | BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                gpu,
            ),
            label: String::from(label),
            capacity,
            next_index: 0,
        }
    }

    pub fn from_labels(label: &str, timestamp_labels: &[&str], gpu: &Arc<Gpu>) -> Self {
        Self::new(label, timestamp_labels.len(), gpu)
    }

    pub fn write_timestamp(
        &mut self,
        command_encoder: &mut CommandEncoder,
    ) -> Result<(), CapacityReachedError> {
        if self.next_index >= self.capacity {
            self.next_index += 1;
            Err(CapacityReachedError {
                capacity: self.capacity,
                index: self.next_index,
            })
        } else {
            command_encoder.write_timestamp(&self.query_set, self.next_index as u32);
            self.next_index += 1;
            Ok(())
        }
    }

    pub fn make_compute_pass_timestamp_writes(&mut self) -> Result<wgpu::ComputePassTimestampWrites, CapacityReachedError> {
        if self.next_index + 1 >= self.capacity {
            self.next_index += 2;
            Err(CapacityReachedError {
                capacity: self.capacity,
                index: self.next_index,
            })
        } else {
            let timestamp_write = wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(self.next_index as u32),
                end_of_pass_write_index: Some((self.next_index as u32) + 1),
            };
            self.next_index += 2;
            Ok(timestamp_write)
        }
    }

    pub fn make_render_pass_timestamp_writes(&mut self) -> Result<wgpu::RenderPassTimestampWrites, CapacityReachedError> {
        if self.next_index + 1 >= self.capacity {
            self.next_index += 2;
            Err(CapacityReachedError {
                capacity: self.capacity,
                index: self.next_index,
            })
        } else {
            let timestamp_write = wgpu::RenderPassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: Some(self.next_index as u32),
                end_of_pass_write_index: Some((self.next_index as u32) + 1),
            };
            self.next_index += 2;
            Ok(timestamp_write)
        }
    }

    pub fn next_index(&mut self) -> Result<u32, CapacityReachedError> {
        if self.next_index >= self.capacity {
            self.next_index += 1;
            Err(CapacityReachedError {
                capacity: self.capacity,
                index: self.next_index,
            })
        } else {
            let index = self.next_index as u32;
            self.next_index += 1;
            Ok(index)
        }
    }

    pub fn resolve(
        &mut self,
        command_encoder: &mut CommandEncoder,
        read_buffer: &MappableBuffer<u64>,
    ) -> Result<(), BufferMapError> {
        command_encoder.resolve_query_set(
            &self.query_set,
            0..(self.next_index as u32),
            self.resolve_buffer.buffer(),
            0,
        );
        self.next_index = 0;
        if read_buffer.is_ready() {
            command_encoder.copy_buffer_to_buffer(
                self.resolve_buffer.buffer(),
                0,
                read_buffer.buffer(),
                0,
                self.resolve_buffer.size(),
            );
            Ok(())
        } else {
            Err(BufferMapError::NotReady)
        }
    }

    pub fn create_resolve_buffer(&self) -> MappableBuffer<u64> {
        MappableBuffer::from_buffer(&self.resolve_buffer)
    }


    pub fn query_set(&self) -> &QuerySet {
        &self.query_set
    }
}

impl Debug for TimeStampQuerySet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimeStampQuerySet {{ label: {}, capacity: {}, next_index: {} }}",
            self.label, self.capacity, self.next_index
        )
    }
}
