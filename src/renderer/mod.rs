pub mod camera;
pub mod context;
pub mod geometry;
pub mod pass;
pub mod settings;
pub mod trivial_volume_renderer;
pub mod volume;

use wasm_bindgen::prelude::*;

use crate::resource;

use crate::renderer::camera::Camera;
use crate::renderer::context::{ContextDescriptor, GPUContext};
use crate::renderer::pass::present_to_screen;
use crate::renderer::pass::{ray_guided_dvr, GPUPass};

use crate::wgsl::create_wgsl_preprocessor;

use bytemuck;
use std::sync::Arc;
use wasm_bindgen::JsCast;
use web_sys::OffscreenCanvas;
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, Buffer, Extent3d, SamplerDescriptor, SubmissionIndex};
use winit::dpi::PhysicalSize;


use crate::framework::event::lifecycle::OnCommandsSubmitted;
use crate::input::Input;
use crate::renderer::pass::present_to_screen::PresentToScreen;
use crate::renderer::pass::ray_guided_dvr::{ChannelSettings, RayGuidedDVR, Resources};
use crate::resource::sparse_residency::texture3d::SparseResidencyTexture3DOptions;
use crate::{MultiChannelVolumeRendererSettings, VolumeDataSource, VolumeManager};
pub use trivial_volume_renderer::TrivialVolumeRenderer;


