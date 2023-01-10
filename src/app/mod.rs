use wgpu::{SubmissionIndex, TextureView};
use winit::event_loop::EventLoop;
use winit::window::Window;
use wgpu_framework::app::GpuApp;
use wgpu_framework::context::{ContextDescriptor, SurfaceContext};
use wgpu_framework::event::window::OnUserEvent;
use wgpu_framework::input::Input;
use crate::event::Event;

pub struct App {}

impl App {
    async fn new() -> Self {
        Self {}
    }
}

impl OnUserEvent for App {
    type UserEvent = Event<()>;

    fn on_user_event(&mut self, event: &Self::UserEvent) {
        todo!()
    }
}

impl GpuApp for App {
    fn init(&mut self, window: &Window, event_loop: &EventLoop<Self::UserEvent>, context: &SurfaceContext) {
        todo!()
    }

    fn render(&mut self, view: &TextureView, input: &Input) -> SubmissionIndex {
        todo!()
    }

    fn get_context_descriptor() -> ContextDescriptor<'static> {
        todo!()
    }
}
