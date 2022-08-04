use std::marker::PhantomData;
use std::ops::RangeBounds;
use std::sync::{Arc, Mutex};
use glam::UVec4;
use crate::renderer::volume::RawVolumeBlock;
use crate::util::extent::{extent_to_uvec, extent_volume, origin_to_uvec};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferAddress, Device, ErrorFilter, Extent3d, ImageCopyTexture, ImageDataLayout, MapMode, Origin3d, Queue, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor};
use crate::GPUContext;

#[readonly::make]
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
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            texture,
            view,
            extent,
            format,
        }
    }

    pub fn create_page_directory(device: &Device, queue: &Queue, extent: Extent3d) -> (Self, Vec<UVec4>) {
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
            },
            bytemuck::cast_slice(data.as_slice())
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        (
            Self {
                texture,
                view,
                extent,
                format,
            },
            data
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

    pub fn data_layout(&self) -> ImageDataLayout {
        let format_info = self.format.describe();
        let width_blocks = self.extent.width / format_info.block_dimensions.0 as u32;
        let height_blocks = self.extent.height / format_info.block_dimensions.1 as u32;
        let byter_per_row = width_blocks * format_info.block_size as u32;
        ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(byter_per_row),
            rows_per_image: std::num::NonZeroU32::new(height_blocks),
        }
    }

    pub fn write<T: bytemuck::Pod>(&self, data: &[T], ctx: &Arc<GPUContext>) {
        ctx.queue.write_texture(
            self.texture.as_image_copy(),
            &bytemuck::cast_slice(data),
            self.data_layout(),
            self.extent,
        );
    }

    // this is done by the backend, but I need to figure out what part is invalid, so I implemented it here as well
    fn validate_texture_copy_range(&self, image_copy_texture: &ImageCopyTexture, copy_size: &Extent3d) -> bool {
        /*
            Let blockWidth be the texel block width of imageCopyTexture.texture.format.
            Let blockHeight be the texel block height of imageCopyTexture.texture.format.
            Let subresourceSize be the imageCopyTexture subresource size of imageCopyTexture.
            Return whether all the conditions below are satisfied:
                (imageCopyTexture.origin.x + copySize.width) ≤ subresourceSize.width
                (imageCopyTexture.origin.y + copySize.height) ≤ subresourceSize.height
                (imageCopyTexture.origin.z + copySize.depthOrArrayLayers) ≤ subresourceSize.depthOrArrayLayers
                copySize.width must be a multiple of blockWidth.
                copySize.height must be a multiple of blockHeight.
             */
        let copy_size = extent_to_uvec(copy_size);
        let block_width = self.format.describe().block_dimensions.0 as u32;
        let block_height = self.format.describe().block_dimensions.1 as u32;
        let image_subresource_origin = origin_to_uvec(&image_copy_texture.origin);
        let subresource_size = extent_to_uvec(&self.extent);
        let mut copy_tex_range_valid = true;
        if !((image_subresource_origin + copy_size).le(&subresource_size)) {
            copy_tex_range_valid = false;
            log::error!("copy size larger than subresource size: {} > {}", image_subresource_origin + copy_size, subresource_size);
        }
        if !(copy_size.x % block_width == 0 && copy_size.y % block_height == 0) {
            copy_tex_range_valid = false;
            log::error!("copy_size not a multiple of block size: {} ({}, {})", copy_size, block_width, block_height);
        }
        copy_tex_range_valid
    }


    // this is done by the backend, but I need to figure out what part is invalid, so I implemented it here as well
    fn validate_linear_texture_data(&self, byte_size: u32, copy_extent: &Extent3d) -> bool {
        /*
        validating linear texture data(layout, byteSize, format, copyExtent)
        Arguments:
            GPUImageDataLayout layout - Layout of the linear texture data.
            GPUSize64 byteSize - Total size of the linear data, in bytes.
            GPUTextureFormat format - Format of the texture.
            GPUExtent3D copyExtent - Extent of the texture to copy.

        Let:
            widthInBlocks be copyExtent.width ÷ the texel block width of format. Assert this is an integer.
            heightInBlocks be copyExtent.height ÷ the texel block height of format. Assert this is an integer.
            bytesInLastRow be widthInBlocks × the size of format.
            Fail if the following input validation requirements are not met:

            If heightInBlocks > 1, layout.bytesPerRow must be specified.
            If copyExtent.depthOrArrayLayers > 1, layout.bytesPerRow and layout.rowsPerImage must be specified.
            If specified, layout.bytesPerRow must be ≥ bytesInLastRow.
            If specified, layout.rowsPerImage must be ≥ heightInBlocks.

        Let:
            bytesPerRow be layout.bytesPerRow ?? 0.
            rowsPerImage be layout.rowsPerImage ?? 0.

        Note: These default values have no effect, as they’re always multiplied by 0.

            Let requiredBytesInCopy be 0.

        If copyExtent.depthOrArrayLayers > 0:
            Increment requiredBytesInCopy by bytesPerRow × rowsPerImage × (copyExtent.depthOrArrayLayers − 1).
        If heightInBlocks > 0:
            Increment requiredBytesInCopy by bytesPerRow × (heightInBlocks − 1) + bytesInLastRow.

        Fail if the following condition is not satisfied:
            The layout fits inside the linear data: layout.offset + requiredBytesInCopy ≤ byteSize.
         */
        let layout = self.data_layout();
        let format = self.format.describe();

        let mut valid = true;

        if !(format.block_dimensions.0 == 1 && format.block_dimensions.1 == 1) {
            log::info!("block dimensions {} {}", format.block_dimensions.0, format.block_dimensions.1);
            // if this got printed than I need to assert that width_in_blocks and height_n_blocks is an integer...
        }
        let width_in_blocks = copy_extent.width % format.block_dimensions.0 as u32;
        let height_in_blocks = copy_extent.height % format.block_dimensions.1 as u32;
        let bytes_in_last_row = width_in_blocks * format.block_size as u32;

        if height_in_blocks > 1 {
            if layout.bytes_per_row.is_none() {
                log::error!("bytes per row must be specified");
                valid = false;
            }
        }
        if copy_extent.depth_or_array_layers > 1 {
            if layout.bytes_per_row.is_none() || layout.rows_per_image.is_none() {
                log::error!("bytes per row and rows per image must be specified");
                valid = false;
            }
        }
        if let Some(bytes_per_row) = layout.bytes_per_row {
            if u32::from(bytes_per_row) < bytes_in_last_row {
                log::error!("bytes per row must be >= bytes in last row");
                valid = false;
            }
        }
        if let Some(rows_per_image) = layout.rows_per_image {
            if u32::from(rows_per_image) < height_in_blocks {
                log::error!("rows per image must be >= height in blocks");
                valid = false;
            }
        }

        let bytes_per_row = if let Some(bytes_per_row) = layout.bytes_per_row {
            u32::from(bytes_per_row)
        } else {
            0
        };
        let rows_per_image = if let Some(rows_per_image) = layout.rows_per_image {
            u32::from(rows_per_image)
        } else {
            0
        };
        let mut required_bytes_in_copy = 0;

        if copy_extent.depth_or_array_layers > 0 {
            required_bytes_in_copy += bytes_per_row * rows_per_image * (copy_extent.depth_or_array_layers - 1);
        }
        if height_in_blocks > 0 {
            required_bytes_in_copy += bytes_per_row * (height_in_blocks - 1) + bytes_in_last_row;
        }

        log::info!("required bytes in copy {}, {}, {}", required_bytes_in_copy, bytes_per_row, rows_per_image);
        if !(layout.offset as u32 + required_bytes_in_copy <= byte_size) {
            log::error!("layout doesn't fit inside linear data");
            valid = false;
        }
        /*

        Fail if the following condition is not satisfied:
            The layout fits inside the linear data: layout.offset + requiredBytesInCopy ≤ byteSize.
         */

        valid
    }

    pub fn write_subregion<T: bytemuck::Pod>(&self, data: &[T], origin: Origin3d, extent: Extent3d, ctx: &Arc<GPUContext>) {
        if origin.x == 0 && origin.y == 0 && origin.z == 0 && extent.width == self.extent.width && extent.height == self.extent.height && extent.depth_or_array_layers == self.extent.depth_or_array_layers {
            self.write(data, ctx);
        } else {
            // todo: validation from here: https://www.w3.org/TR/webgpu/#dom-gpuqueue-writetexture
            // todo: check https://www.w3.org/TR/webgpu/#validating-texture-copy-range
            // todo: check https://www.w3.org/TR/webgpu/#abstract-opdef-validating-linear-texture-data
            let image_copy_texture = ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin,
                aspect: TextureAspect::All,
            };

            if !(
                self.validate_texture_copy_range(&image_copy_texture, &extent) &&
                self.validate_linear_texture_data(data.len() as u32 * 4, &extent)
            ) {
                log::error!("texture copy invalid :(");
            }


            ctx.queue.write_texture(
                image_copy_texture,
                &bytemuck::cast_slice(data),
                self.data_layout(),
                extent,
            );
        }
    }
}

pub async fn map_buffer<S: RangeBounds<BufferAddress>>(buffer: &Buffer, bounds: S) {
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    buffer.slice(bounds)
        .map_async(
            MapMode::Read,
            move |v| {
                sender.send(v)
                    .expect("Could not report buffer mapping");
            });
    receiver.receive()
        .await
        .expect("Could not map buffer")
        .expect("I said: could not map buffer!");
}

#[derive(Eq, PartialEq)]
pub enum BufferState {
    Ready,
    Mapping,
    Mapped,
}

pub struct MappableBufferState {
    state: BufferState,
}

pub struct MappableBuffer<T: bytemuck::Pod> {
    buffer: Buffer,
    state: Arc<Mutex<MappableBufferState>>,
    phantom_data: PhantomData<T>,
}

impl<T: bytemuck::Pod> MappableBuffer<T> {
    pub fn new(buffer: Buffer) -> Self {
        Self {
            buffer,
            state: Arc::new(Mutex::new(MappableBufferState {
                state: BufferState::Ready
            })),
            phantom_data: PhantomData
        }
    }

    pub fn as_buffer_ref(&self) -> &Buffer {
        &self.buffer
    }

    /// Maps a mappable buffer if it is not already mapped or being mapped.
    /// The buffer can be read
    pub fn map_async<S: RangeBounds<BufferAddress>>(&self, mode: MapMode, bounds: S) {
        let s = self.state.clone();
        if self.is_ready() {
            s.lock().unwrap().state = BufferState::Mapping;
            self.buffer
                .slice(bounds)
                .map_async(
                    mode,
                    move |_| {
                        s.lock().unwrap().state = BufferState::Mapped;
                    }
                );
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Ready
    }

    pub fn is_mapped(&self) -> bool {
        self.state.lock().unwrap().state == BufferState::Mapped
    }

    pub fn maybe_read<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> Vec<T> {
        if self.is_mapped() {
            let mapped_range = self.buffer.slice(bounds).get_mapped_range();
            let result: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
            drop(mapped_range);
            result
        } else {
            Vec::new()
        }
    }

    pub fn unmap(&self) {
        if self.is_mapped() {
            self.buffer.unmap();
            self.state.lock().unwrap().state = BufferState::Ready;
        }
    }

    pub fn maybe_read_all(&self) -> Vec<T> {
        let result = self.maybe_read(..);
        self.unmap();
        result
    }
}
