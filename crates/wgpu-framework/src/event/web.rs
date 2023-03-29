use serde::{Deserialize, Serialize};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::{CustomEvent, CustomEventInit, EventTarget, HtmlCanvasElement};
use winit::event_loop::EventLoop;

/// A helper function to register an event handler for custom canvas events (`web_sys::CustomEvent`).
/// Events are deserialized to the given event loop's user event type and dispatched via an event
/// loop proxy, such that they can be received by an `OnUserEvent` listener.
#[cfg(target_arch = "wasm32")]
pub fn register_custom_canvas_event_dispatcher<T: for<'de> Deserialize<'de>>(
    event_name: &str,
    canvas: &HtmlCanvasElement,
    event_loop: &EventLoop<T>,
) {
    let event_loop_proxy = event_loop.create_proxy();
    let handler: Closure<dyn FnMut(JsValue)> =
        Closure::wrap(Box::new(move |canvas_event: JsValue| {
            match canvas_event.dyn_into::<CustomEvent>() {
                Ok(custom_event) => {
                    let event: Result<T, _> = serde_wasm_bindgen::from_value(custom_event.detail());
                    match event {
                        Ok(event) => match event_loop_proxy.send_event(event) {
                            Ok(_) => {}
                            Err(error) => {
                                log::error!("Could not dispatch event: {}", error);
                            }
                        },
                        Err(error) => {
                            log::error!("Could not process event: {}", error);
                        }
                    }
                }
                Err(error) => {
                    log::error!(
                        "Could not cast JsValue to web_sys::CustomEvent: {:?}",
                        error
                    );
                }
            }
        }) as Box<dyn FnMut(JsValue)>);
    canvas
        .add_event_listener_with_callback(event_name, handler.as_ref().unchecked_ref())
        .expect("Could not register canvas event handler");
    handler.forget();
}

#[derive(Debug)]
pub enum EventDispatchError {
    JsValue(JsValue),
    Serde(serde_wasm_bindgen::Error),
}

#[derive(Debug)]
pub enum CanvasEventDispatchError {
    EventDispatch(EventDispatchError),
    Canvas(HtmlCanvasElement),
}

#[cfg(target_arch = "wasm32")]
fn dispatch_custom_event_inner(
    event: Result<CustomEvent, JsValue>,
    event_target: &EventTarget,
) -> Result<bool, EventDispatchError> {
    match event {
        Ok(event) => match event_target.dispatch_event(&event) {
            Ok(result) => Ok(result),
            Err(error) => {
                log::error!("Could not dispatch event: {:?}", error);
                Err(EventDispatchError::JsValue(error))
            }
        },
        Err(error) => {
            log::error!("Could not create custom event: {:?}", error);
            Err(EventDispatchError::JsValue(error))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_event_target_event_with_data<T: Serialize>(
    event_type: &str,
    data: &T,
    event_target: &EventTarget,
) -> Result<bool, EventDispatchError> {
    match serde_wasm_bindgen::to_value(data) {
        Ok(detail) => {
            let mut event_init = CustomEventInit::new();
            event_init.detail(&detail);
            dispatch_custom_event_inner(
                CustomEvent::new_with_event_init_dict(event_type, &event_init),
                event_target,
            )
        }
        Err(error) => {
            log::error!("Could not serialize data: {:?}", error);
            Err(EventDispatchError::Serde(error))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_event_target_event(
    event_type: &str,
    event_target: &EventTarget,
) -> Result<bool, EventDispatchError> {
    dispatch_custom_event_inner(CustomEvent::new(event_type), event_target)
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_canvas_event_with_data<T: Serialize>(
    event_type: &str,
    data: &T,
    canvas: &HtmlCanvasElement,
) -> Result<bool, CanvasEventDispatchError> {
    let canvas_clone = canvas.clone();
    match canvas_clone.dyn_into::<EventTarget>() {
        Ok(event_target) => {
            match dispatch_event_target_event_with_data(event_type, data, &event_target) {
                Ok(result) => Ok(result),
                Err(error) => Err(CanvasEventDispatchError::EventDispatch(error)),
            }
        }
        Err(error) => {
            log::error!("Could not cast canvas into an event target: {:?}", error);
            Err(CanvasEventDispatchError::Canvas(error))
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub fn dispatch_canvas_event(
    event_type: &str,
    canvas: &HtmlCanvasElement,
) -> Result<bool, CanvasEventDispatchError> {
    let canvas_clone = canvas.clone();
    match canvas_clone.dyn_into::<EventTarget>() {
        Ok(event_target) => match dispatch_event_target_event(event_type, &event_target) {
            Ok(result) => Ok(result),
            Err(error) => Err(CanvasEventDispatchError::EventDispatch(error)),
        },
        Err(error) => {
            log::error!("Could not cast canvas into an event target: {:?}", error);
            Err(CanvasEventDispatchError::Canvas(error))
        }
    }
}
