use glam::UVec3;
use crate::util::vec_hash_map::VecHashMap;
use crate::volume::octree::subdivision::VolumeSubdivision;

#[derive(Clone, Debug)]
pub struct ResolutionMapping {
    /// The minimum (i.e., lowest) resolution level in the data set.
    min_resolution: u32,

    /// The maximum (i.e., highest) resolution level in the data set.
    max_resolution: u32,

    /// Maps resolution levels in the octree to their corresponding resolution levels in the data set.
    octree_to_dataset: Vec<usize>,

    /// Maps resolution levels in the data set to one ore more resolution levels in the octree.
    dataset_to_octree: VecHashMap<usize, usize>,
}

impl ResolutionMapping {
    pub fn new(
        octree_subdivisions: &[VolumeSubdivision],
        data_subdivisions: &[UVec3],
        min_lod: u32,
        max_lod: u32,
    ) -> Self {
        // subdivisions are ordered from low to high res
        // data_set_subdivisions are ordered from high res to low res
        // c.min_lod is lowest res for channel
        // c.max_lod is highest res for channel
        let mut octree_to_dataset = Vec::with_capacity(octree_subdivisions.len());
        let mut current_lod = min_lod;
        let mut reached_max_lod = current_lod == max_lod;
        for s in octree_subdivisions.iter() {
            // compare s and ds
            // if s <= ds: collect current_lod
            // else: collect next lod
            while !reached_max_lod
                && s.shape()
                .cmpgt(*data_subdivisions.get(current_lod as usize).unwrap())
                .any()
            {
                current_lod = max_lod.min(current_lod + 1);
                reached_max_lod = current_lod == max_lod;
            }
            octree_to_dataset.push(current_lod as usize);
        }
        let mut dataset_to_octree = VecHashMap::new();
        for (octree_index, dataset_index) in octree_to_dataset.iter().enumerate() {
            dataset_to_octree.insert(*dataset_index, octree_index);
        }

        Self {
            min_resolution: min_lod,
            max_resolution: max_lod,
            octree_to_dataset,
            dataset_to_octree,
        }
    }

    pub fn map_to_dataset_level(&self, octree_level: usize) -> usize {
        *self.octree_to_dataset.get(octree_level).unwrap()
    }

    pub fn map_to_octree_subdivision_level(&self, dataset_level: usize) -> &[usize] {
        self.dataset_to_octree
            .get(&dataset_level)
            .unwrap()
            .as_slice()
    }
}

impl PartialEq for ResolutionMapping {
    fn eq(&self, other: &Self) -> bool {
        self.min_resolution == other.min_resolution && self.max_resolution == other.max_resolution
    }
}
