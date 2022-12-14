use glam::UVec3;

#[derive(Clone, Debug, Default)]
pub struct VolumeSubdivision {
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct OctreeNode {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OctreeLevel {
    shape: UVec3,
    // note: this is redundant since its just shape x*y*z, but we need a u32 anyway for alignment
    num_children: u32,
}

pub fn subdivide(volume_shape: UVec3, target_shape: UVec3) {
    let target_size = 32;
    let target_shape = UVec3::new(target_size,target_size,target_size);
    let input_shape = volume_shape;

    let mut shapes = vec![input_shape];
    let mut subdivisions = Vec::new();
    let mut last_shape = input_shape;
    while last_shape.cmpgt(target_shape).any() {
        let subdivide = vec![
            last_shape.x > last_shape.y / 2 && last_shape.x > last_shape.z / 2,
            last_shape.y > last_shape.x / 2 && last_shape.y > last_shape.z / 2,
            last_shape.z > last_shape.x / 2 && last_shape.z > last_shape.y / 2
        ];

        let shape_raw: Vec<u32> = subdivide.iter().enumerate()
            .map(|(i, &subdivide)| if subdivide { last_shape[i] / 2 } else { last_shape[i] })
            .collect();
        let shape = UVec3::from_slice(shape_raw.as_slice());
        shapes.push(shape);
        subdivisions.push(subdivide);
        last_shape = shape;
    }

    subdivisions.reverse();
    let mut level_shapes = vec![UVec3::ONE];
    let mut nodes = vec![vec![OctreeNode::default()]];
    let mut last_num_children = 1;
    for s in subdivisions {
        let num_children = s.iter().fold(1, |acc, &s| if s { acc * 2 } else { acc });
        level_shapes.push(UVec3::new(
            if s[0] { 2 } else { 1 },
            if s[1] { 2 } else { 1 },
            if s[2] { 2 } else { 1 },
        ) * (level_shapes[level_shapes.len() - 1]));
        last_num_children *= num_children;
        nodes.push(vec![OctreeNode::default(); last_num_children]);
    }
    let num_nodes: Vec<usize> = nodes.iter().map(|n| n.len()).collect();

    log::info!("shapes {:?}", shapes);
    log::info!("nodes {:?}", num_nodes);
    log::info!("levels {:?}", level_shapes);

    //nodes.into_iter().flatten().collect();
}