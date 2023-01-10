use std::sync::Arc;
use wgpu::{SubmissionIndex, SurfaceConfiguration, TextureView};
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
use winit::window::Window;
use wgpu_framework::app::{GpuApp, MapToWindowEvent};
use wgpu_framework::context::{ContextDescriptor, Gpu, SurfaceContext};
use wgpu_framework::event::lifecycle::{OnCommandsSubmitted, PrepareRender, Update};
use wgpu_framework::event::window::{OnResize, OnUserEvent};
use wgpu_framework::input::Input;
use crate::event::Event;

pub struct App {}

impl App {
    async fn new(
        gpu: &Arc<Gpu>,
        surface_configuration: &SurfaceConfiguration,
    ) -> Self {
        Self {}
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

impl OnUserEvent for App {
    type UserEvent = Event<()>;

    fn on_user_event(&mut self, event: &Self::UserEvent) {
        todo!()
    }
}

impl MapToWindowEvent for App {
    fn map_to_window_event(&self, user_event: &Self::UserEvent) -> Option<WindowEvent> {
        match user_event {
            Self::UserEvent::Window(e) => Some(e.clone()),
            _ => None
        }
    }
}

impl PrepareRender for App {
    fn prepare_render(&mut self, input: &Input) {
        todo!()
    }
}

impl Update for App {
    fn update(&mut self, input: &Input) {
        todo!()
    }
}

impl OnCommandsSubmitted for App {
    fn on_commands_submitted(&mut self, input: &Input, submission_index: &SubmissionIndex) {
        todo!()
    }
}

impl OnResize for App {
    fn on_resize(&mut self, width: u32, height: u32) {
        todo!()
    }
}
