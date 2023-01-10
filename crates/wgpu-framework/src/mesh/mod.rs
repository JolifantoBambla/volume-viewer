pub mod vertex;

use crate::geometry::bounds::Bounds3;
use crate::mesh::vertex::{
    FromPositionNormal, FromPositionNormalTextureCoordinates, Position,
};
use crate::util::math::f32::PHI;
use glam::{Vec2, Vec3};
use obj::{load_obj, Obj, ObjError, TexturedVertex};
use std::f32::consts::TAU;

pub struct Mesh<V> {
    name: String,
    aabb: Bounds3,
    faces: Vec<[u32; 3]>,
    vertices: Vec<V>,
}

impl<V> Mesh<V> {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn aabb(&self) -> &Bounds3 {
        &self.aabb
    }
    pub fn faces(&self) -> &Vec<[u32; 3]> {
        &self.faces
    }
    pub fn vertices(&self) -> &Vec<V> {
        &self.vertices
    }
}

impl<V: Position> Mesh<V> {
    pub fn new(name: String, faces: Vec<[u32; 3]>, vertices: Vec<V>) -> Self {
        let points: Vec<Vec3> = vertices.iter().map(|v| v.position()).collect();
        Self {
            name,
            aabb: Bounds3::from(points.as_slice()),
            faces,
            vertices,
        }
    }
}

impl<V: FromPositionNormal + Position> Mesh<V> {
    pub fn new_cube() -> Self {
        // todo: add texture coordinates
        #[rustfmt::skip]
            let vertices = vec![
            // top (0, 0, 1)
            V::from_position_normal(Vec3::new(-1.0, -1.0,  1.0), Vec3::Z,     ),
            V::from_position_normal(Vec3::new( 1.0, -1.0,  1.0), Vec3::Z,     ),
            V::from_position_normal(Vec3::new( 1.0,  1.0,  1.0), Vec3::Z,     ),
            V::from_position_normal(Vec3::new(-1.0,  1.0,  1.0), Vec3::Z,     ),
            // bottom (0, 0, -1)
            V::from_position_normal(Vec3::new(-1.0,  1.0, -1.0), Vec3::Z * -1.),
            V::from_position_normal(Vec3::new( 1.0,  1.0, -1.0), Vec3::Z * -1.),
            V::from_position_normal(Vec3::new( 1.0, -1.0, -1.0), Vec3::Z * -1.),
            V::from_position_normal(Vec3::new(-1.0, -1.0, -1.0), Vec3::Z * -1.),
            // right (1, 0, 0)
            V::from_position_normal(Vec3::new( 1.0, -1.0, -1.0), Vec3::X,     ),
            V::from_position_normal(Vec3::new( 1.0,  1.0, -1.0), Vec3::X,     ),
            V::from_position_normal(Vec3::new( 1.0,  1.0,  1.0), Vec3::X,     ),
            V::from_position_normal(Vec3::new( 1.0, -1.0,  1.0), Vec3::X,     ),
            // left (-1, 0, 0)
            V::from_position_normal(Vec3::new(-1.0, -1.0,  1.0), Vec3::X * -1.),
            V::from_position_normal(Vec3::new(-1.0,  1.0,  1.0), Vec3::X * -1.),
            V::from_position_normal(Vec3::new(-1.0,  1.0, -1.0), Vec3::X * -1.),
            V::from_position_normal(Vec3::new(-1.0, -1.0, -1.0), Vec3::X * -1.),
            // front (0, 1, 0)
            V::from_position_normal(Vec3::new( 1.0,  1.0, -1.0), Vec3::Y,     ),
            V::from_position_normal(Vec3::new(-1.0,  1.0, -1.0), Vec3::Y,     ),
            V::from_position_normal(Vec3::new(-1.0,  1.0,  1.0), Vec3::Y,     ),
            V::from_position_normal(Vec3::new( 1.0,  1.0,  1.0), Vec3::Y,     ),
            // back (0, -1, 0)
            V::from_position_normal(Vec3::new( 1.0, -1.0,  1.0), Vec3::Y * -1.),
            V::from_position_normal(Vec3::new(-1.0, -1.0,  1.0), Vec3::Y * -1.),
            V::from_position_normal(Vec3::new(-1.0, -1.0, -1.0), Vec3::Y * -1.),
            V::from_position_normal(Vec3::new( 1.0, -1.0, -1.0), Vec3::Y * -1.),
        ];

        #[rustfmt::skip]
            let faces: Vec<[u32; 3]> = vec![
            [ 0,  1,  2],
            [ 2,  3,  0], // top
            [ 4,  5,  6],
            [ 6,  7,  4], // bottom
            [ 8,  9, 10],
            [10, 11,  8], // right
            [12, 13, 14],
            [14, 15, 12], // left
            [16, 17, 18],
            [18, 19, 16], // front
            [20, 21, 22],
            [22, 23, 20], // back
        ];
        Self::new("Cube".to_string(), faces, vertices)
    }

    pub fn new_icosahedron() -> Self {
        let inv_phi = 1. / PHI;

        // todo: add texture coordinates
        #[rustfmt::skip]
            let vertex_positions = vec![
            Vec3::new(0., inv_phi, -1.),
            Vec3::new(inv_phi, 1., 0.),
            Vec3::new(-inv_phi, 1., 0.),
            Vec3::new(0., inv_phi, 1.),
            Vec3::new(0., -inv_phi, -1.),
            Vec3::new(-1., 0., inv_phi),
            Vec3::new(0., -inv_phi, -1.),
            Vec3::new( 1.,  0., -inv_phi),
            Vec3::new(1., 0., inv_phi),
            Vec3::new(-1.,  0., -inv_phi),
            Vec3::new(inv_phi, -1., 0.),
            Vec3::new(-inv_phi, -1., 0.),
        ];

        let vertices = vertex_positions
            .iter()
            .map(|&v| V::from_position_normal(v, v.normalize()))
            .collect();

        #[rustfmt::skip]
            let faces = vec![
            [ 2,  1,  0],
            [ 1,  2,  3],
            [ 5,  4,  3],
            [ 4,  8,  3],
            [ 7,  6,  0],
            [ 6,  9,  0],
            [11, 10,  4],
            [10, 11,  6],
            [ 9,  5,  2],
            [ 5,  9, 11],
            [ 8,  7,  1],
            [ 7,  8, 10],
            [ 2,  5,  3],
            [ 8,  1,  3],
            [ 9,  2,  0],
            [ 1,  7,  0],
            [11,  9,  6],
            [ 7, 10,  6],
            [ 5, 11,  4],
            [10,  8,  4],
        ];

        Self::new("Icosahedron".to_string(), faces, vertices)
    }

    pub fn new_icosphere(_num_subdivisions: u32) -> Self {
        todo!()
    }
}

impl<V: FromPositionNormalTextureCoordinates + Position> Mesh<V> {
    pub fn from_obj_source(source: &str) -> Result<Self, ObjError> {
        match load_obj(source.as_bytes()) {
            Ok(obj) => Ok(Self::from_obj(obj)),
            Err(error) => Err(error),
        }
    }

    pub fn from_obj(obj: Obj<TexturedVertex, u32>) -> Self {
        assert_eq!(obj.indices.len() % 3, 0);
        let vertices = obj
            .vertices
            .iter()
            .map(|v| {
                V::from_position_normal_texture_coordinates(
                    Vec3::new(v.position[0], v.position[1], v.position[2]),
                    Vec3::new(v.normal[0], v.normal[1], v.normal[2]),
                    Vec3::new(v.texture[0], v.texture[1], v.texture[2]).truncate(),
                )
            })
            .collect();
        let mut faces = Vec::new();
        for i in 0..obj.indices.len() / 3 {
            let base_index = i * 3;
            faces.push([
                obj.indices[base_index],
                obj.indices[base_index + 1],
                obj.indices[base_index + 2],
            ]);
        }
        Self::new(
            obj.name.unwrap_or_else(|| "unnamed obj".to_string()),
            faces,
            vertices,
        )
    }

    pub fn new_default_cylinder(centered: bool) -> Self {
        Mesh::new_cylinder(32, 1, centered)
    }

    // ported from https://vorg.github.io/pex/docs/pex-gen/Cylinder.html
    // Copyright (c) 2012-2014 Marcin Ignac
    pub fn new_cylinder(num_sides: usize, num_segments: usize, centered: bool) -> Self {
        let radius: f32 = 0.5;
        let height: f32 = 1.0;

        // => radius for top & bottom cap
        let r_top = radius;
        let r_bottom = radius;

        // => generate top & bottom cap
        let bottom_cap = true;
        let top_cap = true;

        let mut faces = Vec::new();
        let mut vertices = Vec::new();

        let mut index = 0;

        let offset_y = if centered { 0.0 } else { -height / 2. };

        for j in 0..num_segments + 1 {
            for i in 0..num_sides + 1 {
                let segment_ratio = j as f32 / num_segments as f32;
                let side_ratio = i as f32 / num_sides as f32;
                let r = r_bottom + (r_top - r_bottom) * segment_ratio;
                let y = offset_y + height * segment_ratio;
                let x = r * f32::cos(side_ratio * TAU);
                let z = r * f32::sin(side_ratio * TAU);
                vertices.push(V::from_position_normal_texture_coordinates(
                    Vec3::new(x, y, z),
                    Vec3::new(x, 0.0, z),
                    Vec2::new(side_ratio, segment_ratio),
                ));
                if i < num_sides && j < num_segments {
                    let i0 = index + 1;
                    let i1 = index;
                    let i2 = index + num_sides as u32 + 1;
                    let i3 = index + num_sides as u32 + 2;
                    faces.push([i0, i1, i2]);
                    faces.push([i0, i2, i3]);
                }
                index += 1;
            }
        }

        if bottom_cap {
            vertices.push(V::from_position_normal_texture_coordinates(
                Vec3::new(0.0, offset_y, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec2::new(0.0, 0.0),
            ));
            let center_index = index;
            index += 1;
            for i in 0..num_sides + 1 {
                let y = offset_y;
                let x = r_bottom * f32::cos((i as f32 / num_sides as f32) * TAU);
                let z = r_bottom * f32::sin((i as f32 / num_sides as f32) * TAU);
                vertices.push(V::from_position_normal_texture_coordinates(
                    Vec3::new(x, y, z),
                    Vec3::new(0.0, -1.0, 0.0),
                    Vec2::new(0.0, 0.0),
                ));
                if i < num_sides {
                    faces.push([index, index + 1, center_index]);
                }
                index += 1;
            }
        }

        if top_cap {
            vertices.push(V::from_position_normal_texture_coordinates(
                Vec3::new(0.0, offset_y + height, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec2::new(0.0, 0.0),
            ));
            let center_index = index;
            index += 1;
            for i in 0..num_sides + 1 {
                let y = offset_y + height;
                let x = r_top * f32::cos((i as f32 / num_sides as f32) * TAU);
                let z = r_top * f32::sin((i as f32 / num_sides as f32) * TAU);
                vertices.push(V::from_position_normal_texture_coordinates(
                    Vec3::new(x, y, z),
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec2::new(1.0, 1.0),
                ));
                if i < num_sides {
                    faces.push([index + 1, index, center_index]);
                }
                index += 1;
            }
        }

        Self::new(
            format!(
                "Cylinder (r={}, h={}, centered={})",
                radius, height, centered
            ),
            faces,
            vertices,
        )
    }
}
