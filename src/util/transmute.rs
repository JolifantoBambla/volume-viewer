use std::sync::Arc;

// helper structs to extract private fields - sorry :/
// todo: ignore dead-code
struct Context(web_sys::Gpu);

#[allow(dead_code)]
#[derive(Clone)]
pub struct Adapter {
    context: Arc<Context>,
    pub id: web_sys::GpuAdapter,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct Device {
    context: Arc<Context>,
    pub id: web_sys::GpuDevice,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct Queue {
    context: Arc<Context>,
    pub id: web_sys::GpuQueue,
}
