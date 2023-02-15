use crate::context::Gpu;
use crate::gpu::buffer::MappableBuffer;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use wgpu::{CommandEncoder, Label, QuerySet, QuerySetDescriptor, QueryType};

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

#[derive(Debug)]
pub struct IncompatibleBufferSizeError {
    capacity: usize,
    buffer_size: usize,
}

impl Display for IncompatibleBufferSizeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TimestampQuerySet has capacity {} but destination buffer only has capacity {}",
            self.capacity, self.buffer_size
        )
    }
}

impl Error for IncompatibleBufferSizeError {}

pub struct TimeStampQuerySet {
    query_set: QuerySet,
    label: String,
    capacity: usize,
    next_index: usize,
}

impl TimeStampQuerySet {
    pub fn new(label: &str, capacity: usize, gpu: &Arc<Gpu>) -> Self {
        Self {
            query_set: gpu.device().create_query_set(&QuerySetDescriptor {
                label: Label::from(label),
                ty: QueryType::Timestamp,
                count: capacity as u32,
            }),
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

    pub fn resolve(
        &mut self,
        command_encoder: &mut CommandEncoder,
        destination: &MappableBuffer<u64>,
    ) -> Result<(), IncompatibleBufferSizeError> {
        self.next_index = 0;
        if destination.num_elements() < self.next_index {
            Err(IncompatibleBufferSizeError {
                capacity: self.capacity,
                buffer_size: destination.num_elements(),
            })
        } else {
            command_encoder.resolve_query_set(
                &self.query_set,
                0..(self.next_index as u32),
                destination.buffer(),
                0,
            );
            Ok(())
        }
    }

    pub fn create_resolve_buffer(&self, gpu: &Arc<Gpu>) -> MappableBuffer<u64> {
        MappableBuffer::new(
            format!("TimeStampQuerySet resolve buffer [{}]", self.label).as_str(),
            self.capacity,
            gpu,
        )
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
