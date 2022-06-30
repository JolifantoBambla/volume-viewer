use crate::renderer::volume::RawVolumeBlock;
use wgpu::util::DeviceExt;
use wgpu::{Device, Extent3d, Queue, TextureView};

#[readonly::make]
pub struct Texture {
    pub view: TextureView,
    pub extent: Extent3d,
}

impl Texture {
    pub fn is_3d(&self) -> bool {
        self.extent.depth_or_array_layers > 1u32
    }

    pub fn create_storage_texture(device: &Device, width: u32, height: u32) -> Self {
        let texture_extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            view,
            extent: texture_extent,
        }
    }

    pub fn create_texture_3d(
        device: &Device,
        queue: &Queue,
        data: &[u8],
        extent: Extent3d,
    ) -> Self {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("Texture3D"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            },
            bytemuck::cast_slice(data),
        );
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            view: texture_view,
            extent,
        }
    }

    pub fn from_raw_volume_block(device: &Device, queue: &Queue, volume: &RawVolumeBlock) -> Self {
        Texture::create_texture_3d(
            device,
            queue,
            volume.data.as_slice(),
            volume.create_extent(),
        )
    }
}
