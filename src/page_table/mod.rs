use std::borrow::BorrowMut;
use std::cell::RefCell;
use crate::renderer::resources::Texture;
use glam::{UVec3, UVec4, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::collections::vec_deque::Drain;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{EventTarget};
use wgpu::{Device, Queue};
use crate::util::extent::{box_volume, extent_to_uvec, uvec_to_extent};

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PageTableEntryFlag {
    Unmapped = 0,
    Mapped = 1,
    Empty = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PageTableEntry {
    /// The 3D texture coordinate of the brick referenced by this `PageTableEntry` in the brick
    /// cache.
    /// Note: this is only valid if `flag` is `PageTableEntryFlag::Mapped`
    pub location: UVec3,

    /// A flag signalling if the brick referenced by this `PageTableEntry` is present (`PageTableEntryFlag::Mapped`),
    /// not present and possible non-empty (`PageTableEntryFlag::Unmapped`), or possibly present but
    /// does not hold any meaningful values w.r.t. the current parameters (e.g. transfer function,
    /// threshold, ...) (`PageTableEntryFlag::Empty`).
    pub flag: PageTableEntryFlag,
}

impl PageTableEntry {
    pub fn new(location: UVec3, flag: PageTableEntryFlag) -> Self {
        Self { location, flag }
    }
}

impl Default for PageTableEntry {
    fn default() -> Self {
        UVec4::default().into()
    }
}

impl From<UVec4> for PageTableEntry {
    fn from(v: UVec4) -> Self {
        Self {
            location: v.truncate(),
            flag: match v.w {
                0 => PageTableEntryFlag::Unmapped,
                1 => PageTableEntryFlag::Mapped,
                2 => PageTableEntryFlag::Empty,
                _ => {
                    log::warn!("got unknown page table entry flag value {}", v.w);
                    PageTableEntryFlag::Unmapped
                }
            },
        }
    }
}

impl Into<UVec4> for PageTableEntry {
    fn into(self) -> UVec4 {
        self.location.extend(self.flag as u32)
    }
}

// todo: move out of here
#[derive(Clone, Copy)]
pub struct VolumeResolutionMeta {
    /// The size of the volume in voxels.
    /// It is not necessarily a multiple of `brick_size`.
    /// See `padded_volume_size`.
    volume_size: UVec3,

    /// The size of the volume in voxels padded s.t. it is a multiple of `PageTableMeta::brick_size`.
    padded_volume_size: UVec3,

    /// The spatial extent of the volume.
    scale: Vec3,
}

// todo: move out of here
#[derive(Clone)]
pub struct MultiResolutionVolumeMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    pub(crate) brick_size: UVec3,

    /// The resolutions
    pub(crate) resolutions: Vec<VolumeResolutionMeta>,
}

impl MultiResolutionVolumeMeta {
    pub fn bricks_per_dimension(&self, level: usize) -> UVec3 {
        self.resolutions[level].padded_volume_size / self.brick_size
    }

    pub fn number_of_bricks(&self, level: usize) -> u32 {
        box_volume(self.bricks_per_dimension(level))
    }
}

#[derive(Clone)]
pub struct PageTableMeta {
    /// The offset of this resolution's page table in the page directory.
    offset: UVec3,

    extent: UVec3,

    ///
    volume_meta: VolumeResolutionMeta,
}

#[derive(Clone)]
pub struct PageDirectoryMeta {
    /// The size of a brick in the brick cache. This is constant across all resolutions of the
    /// bricked multi-resolution volume.
    brick_size: UVec3,

    extent: UVec3,

    /// The resolutions
    resolutions: Vec<PageTableMeta>,
}

// todo: address translation
impl PageDirectoryMeta {
    pub fn new(volume_meta: &MultiResolutionVolumeMeta) -> Self {
        let mut resolutions: Vec<PageTableMeta> = Vec::new();
        for (level, volume_resolution) in volume_meta.resolutions.iter().enumerate() {
            let offset = if level > 0 {
                let last_offset = resolutions[level - 1].offset;
                let last_extent = resolutions[level - 1].extent;

                last_offset
                    + if last_extent.x == last_extent.min_element() {
                        UVec3::new(last_extent.x, 0, 0)
                    } else if last_extent.y == last_extent.min_element() {
                        UVec3::new(0, last_extent.y, 0)
                    } else {
                        UVec3::new(0, 0, last_extent.z)
                    }
            } else {
                UVec3::ZERO
            };
            let extent = volume_meta.bricks_per_dimension(level);
            resolutions.push(PageTableMeta {
                offset,
                extent,
                volume_meta: volume_resolution.clone(),
            });
        }

        let extent = resolutions
            .iter()
            .fold(UVec3::ZERO, |a, b| a.max(b.offset + b.extent));

        Self {
            brick_size: volume_meta.brick_size,
            extent,
            resolutions,
        }
    }
}

//#[derive(Deserialize, Serialize)]
pub struct PageTableAddress {
    location: UVec3,
    level: u32,
}

pub struct Brick {
    data: Vec<u8>,
    min: u8,
    max: u8,
}

pub trait SparseResidencyTexture3DSource {
    fn get_meta(&self) -> &MultiResolutionVolumeMeta;

    fn request_bricks(&mut self, brick_addresses: Vec<PageTableAddress>);

    fn poll_bricks(&mut self, limit: usize) -> Vec<(PageTableAddress, Brick)>;
}

pub struct HtmlEventTargetTexture3DSource {
    volume_meta: MultiResolutionVolumeMeta,
    brick_queue: Rc<RefCell<VecDeque<(PageTableAddress, Brick)>>>,
    event_target: EventTarget,
}

impl HtmlEventTargetTexture3DSource {
    pub fn new(volume_meta: MultiResolutionVolumeMeta, event_target: EventTarget) -> Self {
        let mut brick_queue = Rc::new(RefCell::new(VecDeque::new()));
        let mut receiver = brick_queue.clone();
        let event_callback = Closure::wrap(Box::new(
            move |event| {
                receiver.borrow_mut().as_ref().take().push_back((
                    PageTableAddress {
                        location: Default::default(),
                        level: 0
                    },
                    Brick {
                        data: vec![],
                        min: 0,
                        max: 0
                    }
                ));
            }
        ) as Box<dyn FnMut(JsValue)>);

        event_target
            .add_event_listener_with_callback("brick-loaded", event_callback.as_ref().unchecked_ref())
            .ok();
        event_callback.forget();

        Self {
            volume_meta,
            brick_queue,
            event_target,
        }
    }
}

impl SparseResidencyTexture3DSource for HtmlEventTargetTexture3DSource {
    fn get_meta(&self) -> &MultiResolutionVolumeMeta {
        &self.volume_meta
    }

    fn request_bricks(&mut self, brick_addresses: Vec<PageTableAddress>) {
        let request = web_sys::CustomEvent::new("brick-request").ok().unwrap();
        /*request.init_custom_event_with_can_bubble_and_cancelable_and_detail(
            "loader-request",
            false,
            false,
            &JsValue::from_serde(&requested_bricks).unwrap(),
        );*/
        self.event_target.dispatch_event(&request).ok();
    }

    fn poll_bricks(&mut self, limit: usize) -> Vec<(PageTableAddress, Brick)> {
        self.brick_queue.borrow_mut().as_ref().take().drain(..limit).collect()
    }
}


pub struct ExternalTexture3DSource<'a> {
    volume_meta: MultiResolutionVolumeMeta,
    brick_queue: VecDeque<(PageTableAddress, Brick)>,
    // because it might be boring otherwise!
    request_fun: &'a dyn Fn(Vec<PageTableAddress>),
}

impl<'a> ExternalTexture3DSource<'a> {
    pub fn new(volume_meta: MultiResolutionVolumeMeta, request_fun: &'a dyn Fn(Vec<PageTableAddress>)) -> Self {
        Self {
            volume_meta,
            brick_queue: VecDeque::new(),
            request_fun,
        }
    }
}

impl<'a> SparseResidencyTexture3DSource for ExternalTexture3DSource<'a> {
    fn get_meta(&self) -> &MultiResolutionVolumeMeta {
        &self.volume_meta
    }

    fn request_bricks(&mut self, brick_addresses: Vec<PageTableAddress>) {
        (self.request_fun)(brick_addresses);
    }

    fn poll_bricks(&mut self, limit: usize) -> Vec<(PageTableAddress, Brick)> {
        self.brick_queue.drain(..limit).collect()
    }
}

/// Manages a 3D sparse residency texture.
/// A sparse residency texture is not necessarily present in GPU memory as a whole.
pub struct SparseResidencyTexture3D<T>
where
    T: SparseResidencyTexture3DSource,
{
    meta: PageDirectoryMeta,
    page_directory: Texture,
    brick_cache: Texture,
    brick_usage_buffer: Texture,
    request_buffer: Texture,
    local_brick_cache: HashMap<PageTableAddress, Brick>,
    source: T,
}

impl<T: SparseResidencyTexture3DSource> SparseResidencyTexture3D<T> {
    pub fn new(source: T, device: &Device, queue: &Queue) -> Self {
        let volume_meta = source.get_meta();
        let meta = PageDirectoryMeta::new(volume_meta);

        // 1 page table entry per brick
        let page_directory =
            Texture::create_page_directory(device, queue, uvec_to_extent(meta.extent));

        let brick_cache = Texture::create_brick_cache(device);

        let brick_cache_size = extent_to_uvec(brick_cache.extent);
        let bricks_per_dimension = brick_cache_size / meta.brick_size;

        // 1:1 mapping, 1 timestamp per brick in cache
        let brick_usage_buffer = Texture::create_u32_storage_3d(
            "Usage Buffer".to_string(),
            device,
            queue,
            uvec_to_extent(bricks_per_dimension),
        );

        // 1:1 mapping, 1 timestamp per brick in multi-res volume
        let request_buffer = Texture::create_u32_storage_3d(
            "Request Buffer".to_string(),
            device,
            queue,
            uvec_to_extent(meta.extent),
        );

        Self {
            meta,
            page_directory,
            brick_cache,
            brick_usage_buffer,
            request_buffer,
            local_brick_cache: HashMap::new(),
            source,
        }
    }

    /// Call this after rendering has completed to read back requests & usages
    pub fn post_render(&self) {
        // request bricks
    }

    fn find_unused_bricks(&self) {
        // go through usage buffer and find where timestamp = now
        // for all of those which haven't been used in this
    }

    pub fn add_new_brick(&self) {
        // find location in brick cache where to add
        // write brick to brick_cache
        // write page entry to page_directory
    }

    pub fn request_bricks(&self) {
        //
    }
}
