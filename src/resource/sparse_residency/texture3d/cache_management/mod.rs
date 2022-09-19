pub mod process_requests;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Timestamp {
    pub now: u32,
}

impl Timestamp {
    pub fn new(now: u32) -> Self {
        Self { now }
    }
}
