use std::fmt;
use serde::Deserialize;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::CustomEvent;
use winit::dpi::{LogicalPosition, PhysicalPosition};
use winit::event::{
    DeviceId, ElementState, KeyboardInput, MouseButton, MouseScrollDelta, TouchPhase,
    VirtualKeyCode, WindowEvent,
};

use crate::event::Event;

#[derive(Clone, Debug, Deserialize)]
struct MouseEvent {
    #[serde(rename = "type")]
    type_: String,
    button: i16,
}

impl MouseEvent {
    pub fn type_(&self) -> &str {
        &self.type_
    }
    pub fn button(&self) -> i16 {
        self.button
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CursorEvent {
    #[serde(rename = "type")]
    type_: String,
    #[serde(rename = "clientX")]
    client_x: i32,
    #[serde(rename = "clientY")]
    client_y: i32,
}

impl CursorEvent {
    pub fn type_(&self) -> &str {
        &self.type_
    }
    pub fn client_x(&self) -> i32 {
        self.client_x
    }
    pub fn client_y(&self) -> i32 {
        self.client_y
    }
}

#[derive(Clone, Debug, Deserialize)]
struct WheelEvent {
    #[serde(rename = "type")]
    type_: String,
    #[serde(rename = "deltaX")]
    delta_x: f64,
    #[serde(rename = "deltaY")]
    delta_y: f64,
    #[serde(rename = "deltaMode")]
    delta_mode: u32,
}

impl WheelEvent {
    pub fn type_(&self) -> &str {
        &self.type_
    }
    pub fn delta_x(&self) -> f64 {
        self.delta_x
    }
    pub fn delta_y(&self) -> f64 {
        self.delta_y
    }
    pub fn delta_mode(&self) -> u32 {
        self.delta_mode
    }
}

#[derive(Clone, Debug, Deserialize)]
struct KeyboardEvent {
    #[serde(rename = "type")]
    type_: String,
    #[serde(rename = "key")]
    key: String,
}

impl KeyboardEvent {
    pub fn type_(&self) -> &str {
        &self.type_
    }
    pub fn key(&self) -> &str {
        &self.key
    }
}

#[derive(Debug, Clone)]
pub struct ConversionError {
    event_type: String,
    message: String,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Could not convert \"{}\" event: {}",
            self.event_type, self.message
        )
    }
}

pub fn convert_js_event<T>(js_event: JsValue) -> Result<Event<T>, ConversionError> {
    let event = js_event.clone().unchecked_into::<web_sys::Event>();
    match event.type_().as_str() {
        "mousedown" | "mouseup" => {
            if let Ok(mouse_event) = serde_wasm_bindgen::from_value(js_event) {
                convert_mouse_input_event(mouse_event)
            } else {
                Err(ConversionError {
                    event_type: event.type_(),
                    message: "Could not convert event".to_string(),
                })
            }
        }
        "mousemove" => convert_cursor_moved_event(serde_wasm_bindgen::from_value(js_event).unwrap()),
        "wheel" => convert_mouse_wheel_event(serde_wasm_bindgen::from_value(js_event).unwrap()),
        "keydown" | "keypress" | "keyup" => {
            convert_keyboard_event(serde_wasm_bindgen::from_value(js_event).unwrap())
        }
        "rendersettings" => convert_render_settings_event(event.unchecked_into::<CustomEvent>()),
        _ => Err(ConversionError {
            event_type: event.type_(),
            message: "Unsupported event type".to_string(),
        }),
    }
}

fn convert_mouse_input_event<T>(event: MouseEvent) -> Result<Event<T>, ConversionError> {
    let button = match event.button() {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        _ => {
            return Err(ConversionError {
                event_type: event.type_().to_string(),
                message: "Unsupported mouse button".to_string(),
            })
        }
    };
    let state = match event.type_() {
        "mousedown" => ElementState::Pressed,
        "mouseup" => ElementState::Released,
        _ => {
            return Err(ConversionError {
                event_type: event.type_().to_string(),
                message: "Unsupported event type".to_string(),
            })
        }
    };
    #[allow(deprecated)]
    Ok(Event::Window(WindowEvent::MouseInput {
        device_id: unsafe { DeviceId::dummy() },
        state,
        button,
        modifiers: Default::default(),
    }))
}

fn convert_cursor_moved_event<T>(event: CursorEvent) -> Result<Event<T>, ConversionError> {
    #[allow(deprecated)]
    Ok(Event::Window(WindowEvent::CursorMoved {
        device_id: unsafe { DeviceId::dummy() },
        position: PhysicalPosition {
            x: event.client_x() as f64,
            y: event.client_y() as f64,
        },
        modifiers: Default::default(),
    }))
}

fn convert_mouse_wheel_event<T>(event: WheelEvent) -> Result<Event<T>, ConversionError> {
    let x = -event.delta_x();
    let y = -event.delta_y();

    let delta = match event.delta_mode() {
        1 => MouseScrollDelta::LineDelta(x as f32, y as f32),
        0 => {
            let delta = LogicalPosition::new(x, y).to_physical(1.);
            MouseScrollDelta::PixelDelta(delta)
        }
        _ => {
            return Err(ConversionError {
                event_type: event.type_().to_string(),
                message: "Unsupported delta mode".to_string(),
            })
        }
    };
    #[allow(deprecated)]
    Ok(Event::Window(WindowEvent::MouseWheel {
        device_id: unsafe { DeviceId::dummy() },
        delta,
        phase: TouchPhase::Moved,
        modifiers: Default::default(),
    }))
}

fn convert_keyboard_event<T>(event: KeyboardEvent) -> Result<Event<T>, ConversionError> {
    let state = match event.type_() {
        "keydown" | "keypress" => ElementState::Pressed,
        "keyup" => ElementState::Released,
        _ => {
            return Err(ConversionError {
                event_type: event.type_().to_string(),
                message: "Unsupported event type".to_string(),
            })
        }
    };
    let virtual_keycode = match event.key() {
        "Digit1" => VirtualKeyCode::Key1,
        "Digit2" => VirtualKeyCode::Key2,
        "Digit3" => VirtualKeyCode::Key3,
        "Digit4" => VirtualKeyCode::Key4,
        "Digit5" => VirtualKeyCode::Key5,
        "Digit6" => VirtualKeyCode::Key6,
        "Digit7" => VirtualKeyCode::Key7,
        "Digit8" => VirtualKeyCode::Key8,
        "Digit9" => VirtualKeyCode::Key9,
        "Digit0" => VirtualKeyCode::Key0,
        "KeyA" => VirtualKeyCode::A,
        "KeyB" => VirtualKeyCode::B,
        "KeyC" => VirtualKeyCode::C,
        "KeyD" => VirtualKeyCode::D,
        "KeyE" => VirtualKeyCode::E,
        "KeyF" => VirtualKeyCode::F,
        "KeyG" => VirtualKeyCode::G,
        "KeyH" => VirtualKeyCode::H,
        "KeyI" => VirtualKeyCode::I,
        "KeyJ" => VirtualKeyCode::J,
        "KeyK" => VirtualKeyCode::K,
        "KeyL" => VirtualKeyCode::L,
        "KeyM" => VirtualKeyCode::M,
        "KeyN" => VirtualKeyCode::N,
        "KeyO" => VirtualKeyCode::O,
        "KeyP" => VirtualKeyCode::P,
        "KeyQ" => VirtualKeyCode::Q,
        "KeyR" => VirtualKeyCode::R,
        "KeyS" => VirtualKeyCode::S,
        "KeyT" => VirtualKeyCode::T,
        "KeyU" => VirtualKeyCode::U,
        "KeyV" => VirtualKeyCode::V,
        "KeyW" => VirtualKeyCode::W,
        "KeyX" => VirtualKeyCode::X,
        "KeyY" => VirtualKeyCode::Y,
        "KeyZ" => VirtualKeyCode::Z,
        "Escape" => VirtualKeyCode::Escape,
        "F1" => VirtualKeyCode::F1,
        "F2" => VirtualKeyCode::F2,
        "F3" => VirtualKeyCode::F3,
        "F4" => VirtualKeyCode::F4,
        "F5" => VirtualKeyCode::F5,
        "F6" => VirtualKeyCode::F6,
        "F7" => VirtualKeyCode::F7,
        "F8" => VirtualKeyCode::F8,
        "F9" => VirtualKeyCode::F9,
        "F10" => VirtualKeyCode::F10,
        "F11" => VirtualKeyCode::F11,
        "F12" => VirtualKeyCode::F12,
        "F13" => VirtualKeyCode::F13,
        "F14" => VirtualKeyCode::F14,
        "F15" => VirtualKeyCode::F15,
        "F16" => VirtualKeyCode::F16,
        "F17" => VirtualKeyCode::F17,
        "F18" => VirtualKeyCode::F18,
        "F19" => VirtualKeyCode::F19,
        "F20" => VirtualKeyCode::F20,
        "F21" => VirtualKeyCode::F21,
        "F22" => VirtualKeyCode::F22,
        "F23" => VirtualKeyCode::F23,
        "F24" => VirtualKeyCode::F24,
        "PrintScreen" => VirtualKeyCode::Snapshot,
        "ScrollLock" => VirtualKeyCode::Scroll,
        "Pause" => VirtualKeyCode::Pause,
        "Insert" => VirtualKeyCode::Insert,
        "Home" => VirtualKeyCode::Home,
        "Delete" => VirtualKeyCode::Delete,
        "End" => VirtualKeyCode::End,
        "PageDown" => VirtualKeyCode::PageDown,
        "PageUp" => VirtualKeyCode::PageUp,
        "ArrowLeft" => VirtualKeyCode::Left,
        "ArrowUp" => VirtualKeyCode::Up,
        "ArrowRight" => VirtualKeyCode::Right,
        "ArrowDown" => VirtualKeyCode::Down,
        "Backspace" => VirtualKeyCode::Back,
        "Enter" => VirtualKeyCode::Return,
        "Space" => VirtualKeyCode::Space,
        "Compose" => VirtualKeyCode::Compose,
        "Caret" => VirtualKeyCode::Caret,
        "NumLock" => VirtualKeyCode::Numlock,
        "Numpad0" => VirtualKeyCode::Numpad0,
        "Numpad1" => VirtualKeyCode::Numpad1,
        "Numpad2" => VirtualKeyCode::Numpad2,
        "Numpad3" => VirtualKeyCode::Numpad3,
        "Numpad4" => VirtualKeyCode::Numpad4,
        "Numpad5" => VirtualKeyCode::Numpad5,
        "Numpad6" => VirtualKeyCode::Numpad6,
        "Numpad7" => VirtualKeyCode::Numpad7,
        "Numpad8" => VirtualKeyCode::Numpad8,
        "Numpad9" => VirtualKeyCode::Numpad9,
        "AbntC1" => VirtualKeyCode::AbntC1,
        "AbntC2" => VirtualKeyCode::AbntC2,
        "NumpadAdd" => VirtualKeyCode::NumpadAdd,
        "Quote" => VirtualKeyCode::Apostrophe,
        "Apps" => VirtualKeyCode::Apps,
        "At" => VirtualKeyCode::At,
        "Ax" => VirtualKeyCode::Ax,
        "Backslash" => VirtualKeyCode::Backslash,
        "Calculator" => VirtualKeyCode::Calculator,
        "Capital" => VirtualKeyCode::Capital,
        "Semicolon" => VirtualKeyCode::Semicolon,
        "Comma" => VirtualKeyCode::Comma,
        "Convert" => VirtualKeyCode::Convert,
        "NumpadDecimal" => VirtualKeyCode::NumpadDecimal,
        "NumpadDivide" => VirtualKeyCode::NumpadDivide,
        "Equal" => VirtualKeyCode::Equals,
        "Backquote" => VirtualKeyCode::Grave,
        "Kana" => VirtualKeyCode::Kana,
        "Kanji" => VirtualKeyCode::Kanji,
        "AltLeft" => VirtualKeyCode::LAlt,
        "BracketLeft" => VirtualKeyCode::LBracket,
        "ControlLeft" => VirtualKeyCode::LControl,
        "ShiftLeft" => VirtualKeyCode::LShift,
        "MetaLeft" => VirtualKeyCode::LWin,
        "Mail" => VirtualKeyCode::Mail,
        "MediaSelect" => VirtualKeyCode::MediaSelect,
        "MediaStop" => VirtualKeyCode::MediaStop,
        "Minus" => VirtualKeyCode::Minus,
        "NumpadMultiply" => VirtualKeyCode::NumpadMultiply,
        "Mute" => VirtualKeyCode::Mute,
        "LaunchMyComputer" => VirtualKeyCode::MyComputer,
        "NavigateForward" => VirtualKeyCode::NavigateForward,
        "NavigateBackward" => VirtualKeyCode::NavigateBackward,
        "NextTrack" => VirtualKeyCode::NextTrack,
        "NoConvert" => VirtualKeyCode::NoConvert,
        "NumpadComma" => VirtualKeyCode::NumpadComma,
        "NumpadEnter" => VirtualKeyCode::NumpadEnter,
        "NumpadEquals" => VirtualKeyCode::NumpadEquals,
        "OEM102" => VirtualKeyCode::OEM102,
        "Period" => VirtualKeyCode::Period,
        "PlayPause" => VirtualKeyCode::PlayPause,
        "Power" => VirtualKeyCode::Power,
        "PrevTrack" => VirtualKeyCode::PrevTrack,
        "AltRight" => VirtualKeyCode::RAlt,
        "BracketRight" => VirtualKeyCode::RBracket,
        "ControlRight" => VirtualKeyCode::RControl,
        "ShiftRight" => VirtualKeyCode::RShift,
        "MetaRight" => VirtualKeyCode::RWin,
        "Slash" => VirtualKeyCode::Slash,
        "Sleep" => VirtualKeyCode::Sleep,
        "Stop" => VirtualKeyCode::Stop,
        "NumpadSubtract" => VirtualKeyCode::NumpadSubtract,
        "Sysrq" => VirtualKeyCode::Sysrq,
        "Tab" => VirtualKeyCode::Tab,
        "Underline" => VirtualKeyCode::Underline,
        "Unlabeled" => VirtualKeyCode::Unlabeled,
        "AudioVolumeDown" => VirtualKeyCode::VolumeDown,
        "AudioVolumeUp" => VirtualKeyCode::VolumeUp,
        "Wake" => VirtualKeyCode::Wake,
        "WebBack" => VirtualKeyCode::WebBack,
        "WebFavorites" => VirtualKeyCode::WebFavorites,
        "WebForward" => VirtualKeyCode::WebForward,
        "WebHome" => VirtualKeyCode::WebHome,
        "WebRefresh" => VirtualKeyCode::WebRefresh,
        "WebSearch" => VirtualKeyCode::WebSearch,
        "WebStop" => VirtualKeyCode::WebStop,
        "Yen" => VirtualKeyCode::Yen,
        _ => {
            return Err(ConversionError {
                event_type: event.type_().to_string(),
                message: format!("Unsupported key code: {}", event.key()),
            })
        }
    };
    #[allow(deprecated)]
    Ok(Event::Window(WindowEvent::KeyboardInput {
        device_id: unsafe { DeviceId::dummy() },
        input: KeyboardInput {
            scancode: 0, // charCode is deprecated...
            state,
            virtual_keycode: Some(virtual_keycode),
            modifiers: Default::default(),
        },
        is_synthetic: false,
    }))
}

fn convert_render_settings_event<T>(event: CustomEvent) -> Result<Event<T>, ConversionError> {
    // event.detail.setting & event.detail.value
    let detail = serde_wasm_bindgen::from_value(event.detail());
    if let Ok(detail) = detail {
        Ok(Event::Settings(detail))
    } else {
        Err(ConversionError {
            event_type: "rendersettings".to_string(),
            message: "Invalid render settings event".to_string(),
        })
    }
}
