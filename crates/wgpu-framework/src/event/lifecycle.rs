use crate::input::Input;
use wgpu::SubmissionIndex;

pub trait OnFrameBegin {
    fn on_frame_begin(&mut self);
}

pub trait Update {
    fn update(&mut self, input: &Input);
}

pub trait PrepareRender {
    fn prepare_render(&mut self, input: &Input);
}

pub trait OnCommandsSubmitted {
    fn on_commands_submitted(&mut self, input: &Input, submission_index: &SubmissionIndex);
}

pub trait OnFrameEnd {
    fn on_frame_end(&mut self, input: &Input);
}