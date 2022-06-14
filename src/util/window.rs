use winit::{
    event_loop::EventLoop,
    dpi::PhysicalSize,
    platform::web::WindowBuilderExtWebSys,
    window::{ Window, WindowBuilder }
};
use crate::util;

/// Creates a `Window` and `EventLoop` for an existing HTML canvas element.
pub fn create_window(name: String, canvas_id: String) -> (Window, EventLoop<()>) {
    let canvas = util::web::get_canvas_by_id(canvas_id.as_str());
    let canvas_size = PhysicalSize{width: canvas.width(), height: canvas.height()};

    let builder = WindowBuilder::new()
        .with_title(name.as_str())
        .with_canvas(Some(canvas))
        .with_inner_size(canvas_size);

    let event_loop = EventLoop::new();
    let window = builder.build(&event_loop).unwrap();
    (window, event_loop)
}
