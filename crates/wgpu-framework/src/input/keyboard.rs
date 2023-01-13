use std::collections::HashSet;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

#[derive(Clone, Debug)]
pub struct KeyboardEvent {
    key: VirtualKeyCode,
    state: ElementState,
    other_pressed_keys: HashSet<VirtualKeyCode>,
}

impl KeyboardEvent {
    pub fn key(&self) -> VirtualKeyCode {
        self.key
    }
    pub fn is_pressed(&self) -> bool {
        self.state == ElementState::Pressed
    }
    pub fn is_released(&self) -> bool {
        !self.is_pressed()
    }
    pub fn is_other_key_pressed(&self, key: VirtualKeyCode) -> bool {
        self.other_pressed_keys.contains(&key)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Keyboard {
    pressed_keys: HashSet<VirtualKeyCode>,
}

impl Keyboard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next(&self) -> Self {
        self.clone()
    }

    pub fn is_pressed(&self, key: VirtualKeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    pub fn handle_event(&mut self, event: &WindowEvent) -> Option<KeyboardEvent> {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(virtual_keycode),
                        state,
                        ..
                    },
                ..
            } => {
                match state {
                    ElementState::Pressed => {
                        self.pressed_keys.insert(*virtual_keycode);
                    }
                    ElementState::Released => {
                        self.pressed_keys.remove(virtual_keycode);
                    }
                }
                let keyboard_event = KeyboardEvent {
                    key: *virtual_keycode,
                    state: *state,
                    other_pressed_keys: self.pressed_keys.clone(),
                };
                Some(keyboard_event)
            }
            _ => None,
        }
    }
}
