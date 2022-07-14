use std::fmt;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{CustomEvent, KeyboardEvent, MouseEvent, WheelEvent};
use winit::dpi::{LogicalPosition, PhysicalPosition};
use winit::event::{
    DeviceId, ElementState, KeyboardInput, MouseButton, MouseScrollDelta, TouchPhase,
    VirtualKeyCode, WindowEvent,
};

use crate::event::Event;

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

pub fn convert_js_event<'a, T>(js_event: JsValue) -> Result<Event<'a, T>, ConversionError> {
    let event = js_event.unchecked_into::<web_sys::Event>();
    match event.type_().as_str() {
        "mousedown" | "mouseup" => convert_mouse_input_event(event.unchecked_into::<MouseEvent>()),
        "mousemove" => convert_cursor_moved_event(event.unchecked_into::<MouseEvent>()),
        "wheel" => convert_mouse_wheel_event(event.unchecked_into::<WheelEvent>()),
        "keydown" | "keypress" | "keyup" => {
            convert_keyboard_event(event.unchecked_into::<KeyboardEvent>())
        }
        _ => Err(ConversionError {
            event_type: event.type_(),
            message: "Unsupported event type".to_string(),
        }),
    }
}

pub fn convert_mouse_input_event<'a, T>(
    event: MouseEvent,
) -> Result<Event<'a, T>, ConversionError> {
    let button = match event.button() {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        _ => {
            return Err(ConversionError {
                event_type: event.type_(),
                message: "Unsupported mouse button".to_string(),
            })
        }
    };
    let state = match event.type_().as_str() {
        "mousedown" => ElementState::Pressed,
        "mouseup" => ElementState::Released,
        _ => {
            return Err(ConversionError {
                event_type: event.type_(),
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

pub fn convert_cursor_moved_event<'a, T>(
    event: MouseEvent,
) -> Result<Event<'a, T>, ConversionError> {
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

pub fn convert_mouse_wheel_event<'a, T>(
    event: WheelEvent,
) -> Result<Event<'a, T>, ConversionError> {
    let x = -event.delta_x();
    let y = -event.delta_y();

    let delta = match event.delta_mode() {
        WheelEvent::DOM_DELTA_LINE => MouseScrollDelta::LineDelta(x as f32, y as f32),
        WheelEvent::DOM_DELTA_PIXEL => {
            let delta = LogicalPosition::new(x, y).to_physical(1.);
            MouseScrollDelta::PixelDelta(delta)
        }
        _ => {
            return Err(ConversionError {
                event_type: event.type_(),
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

pub fn convert_keyboard_event<'a, T>(
    event: KeyboardEvent,
) -> Result<Event<'a, T>, ConversionError> {
    let scan_code = match event.key_code() {
        0 => event.char_code(),
        i => i,
    };
    let state = match event.type_().as_str() {
        "keydown" | "keypress" => ElementState::Pressed,
        "keyup" => ElementState::Released,
        _ => {
            return Err(ConversionError {
                event_type: event.type_(),
                message: "Unsupported event type".to_string(),
            })
        }
    };
    let virtual_keycode = match &event.code()[..] {
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
                event_type: event.type_(),
                message: "Unsupported key code".to_string(),
            })
        }
    };
    #[allow(deprecated)]
    Ok(Event::Window(WindowEvent::KeyboardInput {
        device_id: unsafe { DeviceId::dummy() },
        input: KeyboardInput {
            scancode: scan_code,
            state,
            virtual_keycode: Some(virtual_keycode),
            modifiers: Default::default(),
        },
        is_synthetic: false,
    }))
}
