use glam::{BVec3, UVec3};

// todo: this should also store which resolution this maps to
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeSubdivision {
    /// The shape of this subdivision.
    shape: UVec3,
    node_offset: u32,
}

impl VolumeSubdivision {
    pub fn new(shape: UVec3, node_offset: u32) -> Self {
        Self { shape, node_offset }
    }

    pub fn from_input_and_target_shape(input_shape: UVec3, target_shape: UVec3) -> Vec<Self> {
        let mut subdivisions = Vec::new();
        let mut last_shape = input_shape;
        while last_shape.cmpgt(target_shape).any() {
            let subdivide = BVec3::new(
                last_shape.x > last_shape.y / 2 && last_shape.x > last_shape.z / 2,
                last_shape.y > last_shape.x / 2 && last_shape.y > last_shape.z / 2,
                last_shape.z > last_shape.x / 2 && last_shape.z > last_shape.y / 2,
            );
            last_shape = UVec3::new(
                if subdivide.x {
                    last_shape.x / 2
                } else {
                    last_shape.x
                },
                if subdivide.y {
                    last_shape.y / 2
                } else {
                    last_shape.y
                },
                if subdivide.z {
                    last_shape.z / 2
                } else {
                    last_shape.z
                },
            );
            subdivisions.push(subdivide);
        }
        subdivisions.reverse();

        let mut volume_subdivisions = vec![Self::default()];

        for (i, s) in subdivisions.iter().enumerate() {
            volume_subdivisions.push(Self::new(
                UVec3::new(
                    if s.x { 2 } else { 1 },
                    if s.y { 2 } else { 1 },
                    if s.z { 2 } else { 1 },
                ) * volume_subdivisions[i].shape,
                volume_subdivisions[i].next_subdivision_offset(),
            ));
        }
        volume_subdivisions
    }

    pub fn num_nodes(&self) -> u32 {
        self.shape.x * self.shape.y * self.shape.z
    }

    pub fn first_node_index(&self) -> u32 {
        self.node_offset
    }

    pub fn last_node_index(&self) -> u32 {
        self.next_subdivision_offset() - 1
    }

    pub fn next_subdivision_offset(&self) -> u32 {
        self.node_offset + self.num_nodes()
    }

    pub fn shape(&self) -> UVec3 {
        self.shape
    }
}

impl Default for VolumeSubdivision {
    fn default() -> Self {
        Self {
            shape: UVec3::ONE,
            node_offset: 0,
        }
    }
}

pub fn total_number_of_nodes(subdivisions: &Vec<VolumeSubdivision>) -> u32 {
    subdivisions.iter().fold(0, |acc, s| acc + s.num_nodes())
}
