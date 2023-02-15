use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use wgpu::{CommandEncoder, MapMode};
use wgpu_framework::context::Gpu;
use wgpu_framework::gpu::buffer::MappableBuffer;
use wgpu_framework::gpu::query_set::TimeStampQuerySet;

pub struct TimestampQueryHelper {
    query_set: TimeStampQuerySet,
    labels: Vec<String>,
    timings: HashMap<String, Vec<u64>>,
    resolve_buffer_pool: Vec<MappableBuffer<u64>>,
    mapped_buffer_queue: VecDeque<MappableBuffer<u64>>,
    buffer_in_last_submit: Option<MappableBuffer<u64>>,
}

impl TimestampQueryHelper {
    pub fn new(label: &str, labels: &[&str], gpu: &Arc<Gpu>) -> Self {
        let query_set = TimeStampQuerySet::from_labels(label, labels, gpu);
        let mut own_labels = Vec::new();
        let mut timings = HashMap::new();
        for l in labels {
            own_labels.push(l.to_string());
            timings.insert(l.to_string(), Vec::new());
        }

        Self {
            query_set,
            labels: own_labels,
            timings,
            resolve_buffer_pool: Vec::new(),
            mapped_buffer_queue: VecDeque::new(),
            buffer_in_last_submit: None,
        }
    }

    pub fn write_timestamp(&mut self, command_encoder: &mut CommandEncoder) {
        if let Err(error) = self.query_set.write_timestamp(command_encoder) {
            log::error!("could not write timestamp: {}", error);
        };
    }

    pub fn resolve(&mut self, command_encoder: &mut CommandEncoder) {
        if self.buffer_in_last_submit.is_some() {
            panic!("last submit's buffer not processed!");
        }
        let buffer = self
            .resolve_buffer_pool
            .pop()
            .unwrap_or_else(|| self.query_set.create_resolve_buffer());
        self.query_set.resolve(command_encoder, &buffer)
            .expect("Could not copy to readable");
        self.buffer_in_last_submit = Some(buffer);
    }

    pub fn map_buffer(&mut self) {
        if let Some(buffer) = self.buffer_in_last_submit.take() {
            buffer
                .map_async(MapMode::Read, ..)
                .expect("Could not map resolve buffer");
            self.mapped_buffer_queue.push_front(buffer);
        }
    }

    pub fn read_buffer(&mut self) {
        let has_mapped_buffer = if let Some(buffer) = self.mapped_buffer_queue.back() {
            buffer.is_mapped()
        } else {
            false
        };
        if has_mapped_buffer {
            let buffer = self.mapped_buffer_queue.pop_back().unwrap();
            let timestamps = buffer.read_all().expect("Could not read mapped buffer");
            for (i, label) in self.labels.iter().enumerate() {
                self.timings
                    .get_mut(label)
                    .unwrap()
                    .push(*timestamps.get(i).unwrap());
            }
            self.resolve_buffer_pool.push(buffer);
        }
    }
}
