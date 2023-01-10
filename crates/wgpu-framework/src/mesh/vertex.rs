use glam::{Vec2, Vec3};
use std::mem;
use wgpu::{BufferAddress, VertexAttribute, VertexBufferLayout, VertexStepMode};

pub trait BufferLayout: bytemuck::Pod {
    fn buffer_layout<'a>() -> VertexBufferLayout<'a>;
}

pub trait WgslDefinition {
    fn wgsl_definition<'a>() -> &'a str;
}

pub trait Position {
    fn position(&self) -> Vec3;
    fn set_position(&mut self, position: Vec3);
}

pub trait Normal {
    fn normal(&self) -> Vec3;
    fn set_normal(&mut self, normal: Vec3);
}

pub trait TextureCoordinates {
    fn texture_coordinates(&self) -> Vec2;
    fn set_texture_coordinates(&mut self, texture_coordinates: Vec2);
}

pub trait FromPosition {
    fn from_position(position: Vec3) -> Self;
}

pub trait FromPositionNormal {
    fn from_position_normal(position: Vec3, normal: Vec3) -> Self;
}

pub trait FromPositionNormalTextureCoordinates {
    fn from_position_normal_texture_coordinates(
        position: Vec3,
        normal: Vec3,
        texture_coordinates: Vec2,
    ) -> Self;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3,
}

impl Position for Vertex {
    fn position(&self) -> Vec3 {
        self.position
    }
    fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }
}

impl Normal for Vertex {
    fn normal(&self) -> Vec3 {
        self.normal
    }

    fn set_normal(&mut self, normal: Vec3) {
        self.normal = normal;
    }
}

impl FromPositionNormal for Vertex {
    fn from_position_normal(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }
}

impl FromPositionNormalTextureCoordinates for Vertex {
    fn from_position_normal_texture_coordinates(
        position: Vec3,
        normal: Vec3,
        _texture_coordinates: Vec2,
    ) -> Self {
        Self::from_position_normal(position, normal)
    }
}

impl BufferLayout for Vertex {
    fn buffer_layout<'a>() -> VertexBufferLayout<'a> {
        const VERTEX_ATTRIBUTES: [VertexAttribute; 2] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
        ];
        VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TexturedVertex {
    position: Vec3,
    normal: Vec3,
    texture_coordinates: Vec2,
}

impl TexturedVertex {
    pub fn new(position: Vec3, normal: Vec3, texture_coordinates: Vec2) -> Self {
        Self {
            position,
            normal,
            texture_coordinates,
        }
    }
}

impl Position for TexturedVertex {
    fn position(&self) -> Vec3 {
        self.position
    }
    fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }
}

impl Normal for TexturedVertex {
    fn normal(&self) -> Vec3 {
        self.normal
    }

    fn set_normal(&mut self, normal: Vec3) {
        self.normal = normal;
    }
}

impl TextureCoordinates for TexturedVertex {
    fn texture_coordinates(&self) -> Vec2 {
        self.texture_coordinates
    }

    fn set_texture_coordinates(&mut self, texture_coordinates: Vec2) {
        self.texture_coordinates = texture_coordinates;
    }
}

impl FromPositionNormalTextureCoordinates for TexturedVertex {
    fn from_position_normal_texture_coordinates(
        position: Vec3,
        normal: Vec3,
        texture_coordinates: Vec2,
    ) -> Self {
        Self::new(position, normal, texture_coordinates)
    }
}

impl BufferLayout for TexturedVertex {
    fn buffer_layout<'a>() -> VertexBufferLayout<'a> {
        const VERTEX_ATTRIBUTES: [VertexAttribute; 3] = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x2
        ];
        VertexBufferLayout {
            array_stride: mem::size_of::<TexturedVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }
    }
}

impl WgslDefinition for TexturedVertex {
    fn wgsl_definition<'a>() -> &'a str {
        include_str!("base_vertex.wgsl")
    }
}
