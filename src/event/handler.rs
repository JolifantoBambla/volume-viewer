use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;
use winit::event_loop::EventLoop;

use crate::event::conversion::convert_js_event;
use crate::event::Event;

pub fn register_window_event_handler(
    canvas: &HtmlCanvasElement,
    event_name: &str,
    handler: &Closure<dyn FnMut(JsValue)>,
) {
    canvas
        .add_event_listener_with_callback(event_name, handler.as_ref().unchecked_ref())
        .unwrap();
}

pub fn register_window_event_handlers(
    canvas: &HtmlCanvasElement,
    handler: &Closure<dyn FnMut(JsValue)>,
) {
    register_window_event_handler(canvas, "mousedown", handler);
    register_window_event_handler(canvas, "mouseup", handler);
    register_window_event_handler(canvas, "mousemove", handler);
    register_window_event_handler(canvas, "wheel", handler);
    register_window_event_handler(canvas, "keydown", handler);
    register_window_event_handler(canvas, "keypress", handler);
    register_window_event_handler(canvas, "keyup", handler);
}

pub fn create_default_js_event_handler<T>(
    event_loop: &EventLoop<Event<T>>,
) -> Closure<dyn FnMut(JsValue)> {
    let event_loop_proxy = event_loop.create_proxy();
    Closure::wrap(Box::new(move |event| {
        let event = convert_js_event(event);
        match event {
            Ok(event) => {
                event_loop_proxy.send_event(event).ok();
            }
            Err(error) => {
                log::error!("Could not dispatch event: {}", error);
            }
        }
    }) as Box<dyn FnMut(JsValue)>)
}

pub fn register_default_js_event_handlers(
    canvas: &HtmlCanvasElement,
    event_loop: &EventLoop<Event<()>>,
) {
    let closure = create_default_js_event_handler(event_loop);
    register_window_event_handlers(canvas, &closure);
    closure.forget();
}
