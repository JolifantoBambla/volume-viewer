use winit::event::WindowEvent;

pub mod conversion;
pub mod handler;

pub enum Event<'a, T: 'static> {
    Window(WindowEvent<'a>),
    Custom(T),
}
