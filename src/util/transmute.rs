use std::sync::Arc;

// std::mem::transmute_copy doesn't work - don't know why - this does, just doesn't work with references:
macro_rules! transmute_copy {
    ($src:expr, $dst_type:ty) => {
        unsafe {
            let transmuted: $dst_type = std::mem::transmute($src);
            $src = std::mem::transmute(transmuted.clone());
            transmuted
        }
    };
}

pub(crate) use transmute_copy;

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
