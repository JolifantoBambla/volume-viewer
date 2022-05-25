use serde::{Serialize, Deserialize};
use serde_json::Value;

use crate::axis::Axis;
use crate::coordinate_transformations::{
    CoordinateTransformation,
    Scale,
    Translation,
};
use crate::util::warn_unless;

#[derive(Serialize, Deserialize)]
pub struct Dataset {
    pub path: String,

    #[serde(rename = "coordinateTransformations")]
    pub coordinate_transformations: Vec<CoordinateTransformation>,
}

#[derive(Serialize, Deserialize)]
pub struct Multiscale {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downscaling_type: Option<String>,

    pub axes: Vec<Axis>,

    // ordered by largest (i.e. highest resolution) to smallest.
    pub datasets: Vec<Dataset>,

    // are applied after `coordinate_transformations` in `datasets`
    #[serde(rename = "coordinateTransformations")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_transformations: Option<Vec<CoordinateTransformation>>,

    // fields in metadata depend on `downscaling_type`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Multiscale {
    pub fn is_valid(&self) -> bool {
        let axes_length = self.axes.len();
        Multiscale::are_multi_scale_axes_valid(&self.axes) &&
            Multiscale::are_coordinate_transformations_valid(self.coordinate_transformations.as_ref().unwrap_or(&Vec::new()), axes_length) &&
            self.datasets.iter().all(|d| Multiscale::are_coordinate_transformations_valid(&d.coordinate_transformations, axes_length))
    }

    fn is_multi_scale_axes_length_valid(axes: &Vec<Axis>) -> bool {
        warn_unless!(
            axes.len() >= 2 && axes.len() <= 5,
            "The spec states: The length of \"axes\" must be between 2 and 5. Got: {}",
            axes.len(),
        )
    }

    fn are_multi_scale_axes_types_valid(axes: &Vec<Axis>) -> bool {
        let space_count = Axis::get_space_axes(axes).len();
        let time_count = Axis::get_time_axes(axes).len();
        let channel_count = Axis::get_channel_axes(axes).len();
        let custom_count = Axis::get_custom_axes(axes).len();
        warn_unless!(
            space_count == 2 || space_count == 3 &&
            time_count <= 1 &&
            (
                (channel_count == 0 && custom_count == 0) ||
                (channel_count == 1 && custom_count == 0) ||
                (channel_count == 0 && custom_count == 1) ||
                !(channel_count == 1 && custom_count == 1)
            ),
            "The spec states: The \"axes\" MUST contain 2 or 3 entries of \"type:space\" and MAY contain one additional entry of \"type:time\" and MAY contain one additional entry of \"type:channel\" or a null / custom type. Got (space,time,channel,custom): ({},{},{},{})",
            space_count,
            time_count,
            channel_count,
            custom_count,
        )
    }

    fn is_multi_scale_axes_order_valid(axes: &Vec<Axis>) -> bool {
        let mut time_indices: Vec<usize> = Vec::new();
        let mut channel_indices: Vec<usize> = Vec::new();
        let mut custom_indices: Vec<usize> = Vec::new();
        let mut space_indices: Vec<usize> = Vec::new();
        for (i, a) in axes.iter().enumerate() {
            match a {
                Axis::Space(_) => {
                    space_indices.push(i);
                },
                Axis::Time(_) => {
                    time_indices.push(i);
                },
                Axis::Channel(_) => {
                    channel_indices.push(i);
                },
                Axis::Custom(_) => {
                    custom_indices.push(i);
                }
            };
        }
        warn_unless!(
            space_indices.iter().all(|&si| {
                channel_indices.iter().all(|&ci| ci < si) &&
                custom_indices.iter().all(|&ci| ci < si)
            }) &&
            time_indices.iter().all(|&ti| {
                channel_indices.iter().all(|&ci| ci < ti) &&
                custom_indices.iter().all(|&ci| ci < ti)
            }),
            "The spec states: the entries MUST be ordered by \"type\" where the \"time\" axis must come first (if present), followed by the \"channel\" or custom axis (if present) and the axes of type \"space\". Got indices (time,channel,custom,space): ({:?},{:?},{:?},{:?})",
            time_indices,
            channel_indices,
            custom_indices,
            space_indices,
        )
    }

    fn are_multi_scale_axes_valid(axes: &Vec<Axis>) -> bool {
        Multiscale::is_multi_scale_axes_length_valid(axes) &&
            Multiscale::are_multi_scale_axes_types_valid(axes) &&
            Multiscale::is_multi_scale_axes_order_valid(axes)
    }

    fn are_coordinate_transformations_dimensions_valid(coordinate_transformations: &Vec<CoordinateTransformation>, axes_length: usize) -> bool {
        coordinate_transformations.iter().all(|c| {
            let transformation_length = match c {
                CoordinateTransformation::Translation(Translation::Translation(translation)) => {
                    translation.len()
                },
                CoordinateTransformation::Scale(Scale::Scale(scale)) => {
                    scale.len()
                },
                _ => {
                    // can't validate "path"
                    axes_length
                }
            };
            warn_unless!(
                transformation_length == axes_length,
                "The spec states: The length of the scale and translation array MUST be the same as the length of \"axes\". Got (axes,transformation): ({},{})",
                transformation_length,
                axes_length
            )
        })
    }

    fn are_coordinate_transformation_types_valid(coordinate_transformations: &Vec<CoordinateTransformation>) -> bool {
        coordinate_transformations.iter().all(|c| {
            warn_unless!(
                match c {
                    CoordinateTransformation::Translation(_) | CoordinateTransformation::Scale(_) => {
                        true
                    },
                    _ => false
                },
                "The spec states: The transformation MUST only be of type translation or scale."
            )
        })
    }

    fn coordinate_transformations_contain_exactly_one_scale(coordinate_transformations: &Vec<CoordinateTransformation>) -> bool {
        let scale_count = coordinate_transformations.iter()
            .filter(|c| matches!(c, CoordinateTransformation::Scale(_)))
            .count();
        warn_unless!(
            scale_count == 1,
            "The spec states: They MUST contain exactly one scale transformation that specifies the pixel size in physical units or time duration. Got: {}",
            scale_count
        )
    }

    fn coordinate_transformations_contain_at_most_one_translation(coordinate_transformations: &Vec<CoordinateTransformation>) -> bool {
        let translation_count = coordinate_transformations.iter()
            .filter(|c| matches!(c, CoordinateTransformation::Translation(_)))
            .count();
        warn_unless!(
            translation_count <= 1,
            "The spec states: It MAY contain exactly one translation that specifies the offset from the origin in physical units. Got: {}",
            translation_count
        )
    }

    fn scale_is_first_element_in_coordinate_transformations(coordinate_transformations: &Vec<CoordinateTransformation>) -> bool {
        warn_unless!(
            match coordinate_transformations[0] {
                CoordinateTransformation::Scale(_) => true,
                _ => false
            },
            "The spec states: If translation is given it MUST be listed after scale to ensure that it is given in physical coordinates."
        )
    }

    fn are_coordinate_transformations_valid(coordinate_transformations: &Vec<CoordinateTransformation>, axes_length: usize) -> bool {
        Multiscale::are_coordinate_transformations_dimensions_valid(coordinate_transformations, axes_length) &&
            Multiscale::are_coordinate_transformation_types_valid(coordinate_transformations) &&
            Multiscale::coordinate_transformations_contain_exactly_one_scale(coordinate_transformations)  &&
            Multiscale::coordinate_transformations_contain_at_most_one_translation(coordinate_transformations) &&
            Multiscale::scale_is_first_element_in_coordinate_transformations(coordinate_transformations)
    }
}