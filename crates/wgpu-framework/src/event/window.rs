use winit::event::WindowEvent;

pub trait OnResize {
    fn on_resize(&mut self, width: u32, height: u32);
}

pub trait OnWindowEvent {
    fn on_window_event(&mut self, event: &WindowEvent);
}

pub trait OnUserEvent {
    type UserEvent;

    fn on_user_event(&mut self, event: &Self::UserEvent);
}
