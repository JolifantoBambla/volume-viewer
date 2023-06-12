use crate::renderer::volume::RawVolumeBlock;
use crate::util::extent::extent_volume;
use glam::UVec4;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use wgpu::{
    Device, Extent3d, ImageCopyTexture, ImageDataLayout, Origin3d, Queue, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};
use wgpu_framework::context::Gpu;

#[readonly::make]
#[derive(Debug)]
pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: TextureView,
    pub extent: Extent3d,
    pub format: TextureFormat,
}

impl Texture {
    pub fn is_3d(&self) -> bool {
        self.extent.depth_or_array_layers > 1u32
    }

    pub fn create_storage_texture(device: &Device, width: u32, height: u32) -> Self {
        let format = TextureFormat::Rgba8Unorm;
        let extent = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[format],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
            format,
        }
    }

    pub fn create_page_directory(
        device: &Device,
        queue: &Queue,
        extent: Extent3d,
    ) -> (Self, Vec<UVec4>) {
        let data = vec![UVec4::ZERO; extent_volume(&extent) as usize];
        let format = TextureFormat::Rgba32Uint;
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Page Directory"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[format],
            },
            bytemuck::cast_slice(data.as_slice()),
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        (
            Self {
                texture,
                view,
                extent,
                format,
            },
            data,
        )
    }

    pub fn create_brick_cache(device: &Device, extent: Extent3d) -> Self {
        let format = TextureFormat::R8Unorm;
        let max_texture_dimension = device.limits().max_texture_dimension_3d;
        assert!(
            extent.width <= max_texture_dimension
                && extent.height <= max_texture_dimension
                && extent.depth_or_array_layers <= max_texture_dimension,
            "Brick cache extent must not exceed device limits"
        );
        //device.push_error_scope(ErrorFilter::OutOfMemory);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("Brick Cache"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[format],
        });
        //device.pop_error_scope();
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
            format,
        }
    }

    pub fn create_u32_storage_3d(
        label: String,
        device: &Device,
        queue: &Queue,
        extent: Extent3d,
    ) -> Self {
        let format = TextureFormat::R32Uint;
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some(label.as_str()),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::STORAGE_BINDING,
                view_formats: &[format],
            },
            vec![0u8; (extent_volume(&extent) * 4) as usize].as_slice(),
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
            format,
        }
    }

    pub fn create_texture_3d(
        device: &Device,
        queue: &Queue,
        data: &[u8],
        extent: Extent3d,
    ) -> Self {
        let format = TextureFormat::R8Unorm;
        let texture = device.create_texture_with_data(
            queue,
            &TextureDescriptor {
                label: Some("Texture3D"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[format],
            },
            bytemuck::cast_slice(data),
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
            format,
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

    pub fn num_pixels(&self) -> usize {
        (self.extent.width * self.extent.height * self.extent.depth_or_array_layers) as usize
    }

    pub fn data_layout(&self, extent: &Extent3d) -> ImageDataLayout {
        let physical_extent = extent.physical_size(self.format);
        let block_dimensions = self.format.block_dimensions();
        let block_size = self.format.block_size(None).expect("Could not get block size");
        let width_blocks = physical_extent.width / block_dimensions.0 as u32;
        let height_blocks = physical_extent.height / block_dimensions.1 as u32;
        let bytes_per_row = width_blocks * block_size as u32;
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height_blocks),
        }
    }

    pub fn write<T: bytemuck::Pod>(&self, data: &[T], ctx: &Arc<Gpu>) {
        ctx.queue().write_texture(
            self.texture.as_image_copy(),
            bytemuck::cast_slice(data),
            self.data_layout(&self.extent),
            self.extent,
        );
    }

    fn is_whole_texture(&self, origin: &Origin3d, extent: &Extent3d) -> bool {
        origin.x == 0
            && origin.y == 0
            && origin.z == 0
            && extent.width == self.extent.width
            && extent.height == self.extent.height
            && extent.depth_or_array_layers == self.extent.depth_or_array_layers
    }

    pub fn write_subregion<T: bytemuck::Pod>(
        &self,
        data: &[T],
        origin: Origin3d,
        extent: Extent3d,
        ctx: &Arc<Gpu>,
    ) {
        if self.is_whole_texture(&origin, &extent) {
            self.write(data, ctx);
        } else {
            let image_copy_texture = ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin,
                aspect: TextureAspect::All,
            };

            ctx.queue().write_texture(
                image_copy_texture,
                bytemuck::cast_slice(data),
                self.data_layout(&extent),
                extent,
            );
        }
    }
}
