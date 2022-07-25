use crate::renderer::volume::RawVolumeBlock;
use wgpu::util::DeviceExt;
use wgpu::{
    Device, Extent3d, Queue, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};

#[readonly::make]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: TextureView,
    pub extent: Extent3d,
}

impl Texture {
    pub fn is_3d(&self) -> bool {
        self.extent.depth_or_array_layers > 1u32
    }

    pub fn create_storage_texture(device: &Device, width: u32, height: u32) -> Self {
        let texture_extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent: texture_extent,
        }
    }

    pub fn create_page_directory(device: &Device, queue: &Queue, extent: Extent3d) -> Self {
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Page Directory"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            },
            bytemuck::cast_slice(
                vec![
                    0u8;
                    (extent.width * extent.height * extent.depth_or_array_layers * 4) as usize
                ]
                .as_slice(),
            ),
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
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
            &TextureDescriptor {
                label: Some("Texture3D"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            },
            bytemuck::cast_slice(data),
        );
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
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
