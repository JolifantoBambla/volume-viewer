use winit::event::WindowEvent;

pub mod conversion;
pub mod handler;

pub struct RawArrayReceived {
    pub data: Vec<u16>,
    pub shape: Vec<u32>,
}

pub enum Event<'a, T: 'static> {
    Window(WindowEvent<'a>),
    RawArray(RawArrayReceived),
    Custom(T),
}
