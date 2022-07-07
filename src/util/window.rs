use crate::util;
use web_sys::HtmlCanvasElement;
use winit::{
    dpi::PhysicalSize,
    event_loop::EventLoop,
    platform::web::WindowBuilderExtWebSys,
    window::{Window, WindowBuilder},
};

/// Creates a `Window` and `EventLoop` for an existing HTML canvas element.
pub fn create_window(name: String, canvas_id: String) -> (Window, EventLoop<()>) {
    let canvas = util::web::get_canvas_by_id(canvas_id.as_str());
    create_window_from_canvas(name, canvas)
}

pub fn create_window_from_canvas(name: String, canvas: HtmlCanvasElement) -> (Window, EventLoop<()>) {
    let canvas_size = PhysicalSize {
        width: canvas.width(),
        height: canvas.height(),
    };

    let builder = window_builder_without_size(name, canvas)
        .with_inner_size(canvas_size);

    let event_loop = EventLoop::new();
    let window = builder.build(&event_loop).unwrap();
    (window, event_loop)
}

pub fn window_builder_without_size(name: String, canvas: HtmlCanvasElement) -> WindowBuilder {
    WindowBuilder::new()
        .with_title(name.as_str())
        .with_canvas(Some(canvas))
}