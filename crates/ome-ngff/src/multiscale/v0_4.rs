use serde::{Serialize, Deserialize};
use serde_json::Value;

use crate::axis::{
    Axis,
    ChannelAxis,
    CustomAxis,
    SpaceAxis,
    TimeAxis,
};
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
    pub fn get_space_axes(&self) -> Vec<SpaceAxis> {
        self.axes
            .iter()
            .filter(|&a| matches!(a, Axis::Space(_)))
            .map(|a| a.clone_as_space_axis().unwrap())
            .collect()
    }

    pub fn get_time_axes(&self) -> Vec<TimeAxis> {
        self.axes
            .iter()
            .filter(|&a| matches!(a, Axis::Time(_)))
            .map(|a| a.clone_as_time_axis().unwrap())
            .collect()
    }

    pub fn get_channel_axes(&self) -> Vec<ChannelAxis> {
        self.axes
            .iter()
            .filter(|&a| matches!(a, Axis::Channel(_)))
            .map(|a| a.clone_as_channel_axis().unwrap())
            .collect()
    }

    pub fn get_custom_axes(&self) -> Vec<CustomAxis> {
        self.axes
            .iter()
            .filter(|&a| matches!(a, Axis::Custom(_)))
            .map(|a| a.clone_as_custom_axis().unwrap())
            .collect()
    }

    pub fn get_time_axis(&self) -> Option<TimeAxis> {
        match &self.axes[0] {
            Axis::Time(axis) => Some(axis.clone()),
            _ => None
        }
    }

    pub fn get_channel_axis(&self) -> Option<ChannelAxis> {
        let channel_axes: Vec<ChannelAxis> = self.get_channel_axes();
        if channel_axes.is_empty() {
            None
        } else {
            Some(channel_axes[0].clone())
        }
    }

    pub fn get_custom_axis(&self) -> Option<CustomAxis> {
        let custom_axes: Vec<CustomAxis> = self.get_custom_axes();
        if custom_axes.is_empty() {
            None
        } else {
            Some(custom_axes[0].clone())
        }
    }

    pub fn is_valid(&self) -> bool {
        let axes_length = self.axes.len();
        self.are_axes_valid() &&
            Multiscale::are_coordinate_transformations_valid(self.coordinate_transformations.as_ref().unwrap_or(&Vec::new()), axes_length) &&
            self.datasets.iter().all(|d| Multiscale::are_coordinate_transformations_valid(&d.coordinate_transformations, axes_length))
    }

    fn is_axes_length_valid(&self) -> bool {
        warn_unless!(
            self.axes.len() >= 2 && self.axes.len() <= 5,
            "The spec states: The length of \"axes\" must be between 2 and 5. Got: {}",
            self.axes.len(),
        )
    }

    fn are_axes_types_valid(&self) -> bool {
        let space_count = self.get_space_axes().len();
        let time_count = self.get_time_axes().len();
        let channel_count = self.get_channel_axes().len();
        let custom_count = self.get_custom_axes().len();
        warn_unless!(
            space_count == 2 || space_count == 3 &&
            time_count <= 1 &&
            channel_count <= 1 &&
            custom_count <= 1 &&
            channel_count + custom_count <= 1,
            "The spec states: The \"axes\" MUST contain 2 or 3 entries of \"type:space\" and MAY contain one additional entry of \"type:time\" and MAY contain one additional entry of \"type:channel\" or a null / custom type. Got (space,time,channel,custom): ({},{},{},{})",
            space_count,
            time_count,
            channel_count,
            custom_count,
        )
    }

    fn is_axes_order_valid(&self) -> bool {
        let mut time_indices: Vec<usize> = Vec::new();
        let mut channel_indices: Vec<usize> = Vec::new();
        let mut custom_indices: Vec<usize> = Vec::new();
        let mut space_indices: Vec<usize> = Vec::new();
        for (i, a) in self.axes.iter().enumerate() {
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
                channel_indices.iter().all(|&ci| ti < ci) &&
                custom_indices.iter().all(|&ci| ti < ci)
            }),
            "The spec states: the entries MUST be ordered by \"type\" where the \"time\" axis must come first (if present), followed by the \"channel\" or custom axis (if present) and the axes of type \"space\". Got indices (time,channel,custom,space): ({:?},{:?},{:?},{:?})",
            time_indices,
            channel_indices,
            custom_indices,
            space_indices,
        )
    }

    fn are_axes_valid(&self) -> bool {
        self.is_axes_length_valid() &&
            self.are_axes_types_valid() &&
            self.is_axes_order_valid()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn t() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "t", "type": "time", "unit": "millisecond"}"#.to_string()).unwrap()
    }
    fn c() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "c", "type": "channel"}"#.to_string()).unwrap()
    }
    fn custom() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "foo"}"#.to_string()).unwrap()
    }
    fn z() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "z", "type": "space", "unit": "micrometer"}"#.to_string()).unwrap()
    }
    fn y() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "y", "type": "space", "unit": "micrometer"}"#.to_string()).unwrap()
    }
    fn x() -> Axis {
        serde_json::from_str::<Axis>(&r#"{"name": "x", "type": "space", "unit": "micrometer"}"#.to_string()).unwrap()
    }
    fn tczyx() -> Vec<Axis> {
        vec![t(), c(), z(), y(), x()]
    }
    fn t_custom_zyx() -> Vec<Axis> {
        vec![t(), custom(), z(), y(), x()]
    }
    fn czyx() -> Vec<Axis> {
        vec![c(), z(), y(), x()]
    }
    fn custom_zyx() -> Vec<Axis> {
        vec![custom(), z(), y(), x()]
    }
    fn zyx() -> Vec<Axis> {
        vec![z(), y(), x()]
    }
    fn tcyx() -> Vec<Axis> {
        vec![t(), c(), y(), x()]
    }
    fn t_custom_yx() -> Vec<Axis> {
        vec![t(), custom(), y(), x()]
    }
    fn cyx() -> Vec<Axis> {
        vec![c(), y(), x()]
    }
    fn custom_yx() -> Vec<Axis> {
        vec![custom(), y(), x()]
    }
    fn yx() -> Vec<Axis> {
        vec![y(), x()]
    }
    fn tzyx() -> Vec<Axis> {
        vec![t(), z(), y(), x()]
    }
    fn tyx() -> Vec<Axis> {
        vec![t(), y(), x()]
    }
    fn multiscale_from_axes(axes: Vec<Axis>) -> Multiscale {
        Multiscale{
            name: None,
            downscaling_type: None,
            axes,
            datasets: vec![],
            coordinate_transformations: None,
            metadata: None
        }
    }

    fn spec_example_json() -> String {
        r#"{
            "version": "0.4",
            "name": "example",
            "axes": [
                {"name": "t", "type": "time", "unit": "millisecond"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 0.5, 0.5, 0.5]
                    }]
                },
                {
                    "path": "1",
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0]
                    }]
                },
                {
                    "path": "2",
                    "coordinateTransformations": [{
                        "type": "scale",
                        "scale": [1.0, 1.0, 2.0, 2.0, 2.0]
                    }]
                }
            ],
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [0.1, 1.0, 1.0, 1.0, 1.0]
            }],
            "type": "gaussian",
            "metadata": {
                "description": "the fields in metadata depend on the downscaling implementation. Here, the parameters passed to the skimage function are given",
                "method": "skimage.transform.pyramid_gaussian",
                "version": "0.16.1",
                "args": "[true]",
                "kwargs": {"multichannel": true}
            }
        }"#.to_string()
    }

    #[test]
    fn spec_example_is_valid() {
        assert!(
            serde_json::from_str::<Multiscale>(&spec_example_json()).unwrap().is_valid()
        );
    }

    #[test]
    fn test_invalid_empty_axes() {
        let multiscale = multiscale_from_axes(vec![]);
        assert!(!multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_invalid_too_many_axes() {
        let multiscale = multiscale_from_axes(vec![
            t(),
            c(),
            custom(),
            z(),
            y(),
            x()
        ]);
        assert!(!multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_valid_axes_length_tczyx() {
        let multiscale = multiscale_from_axes(tczyx());
        assert!(multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_valid_axes_length_czyx() {
        let multiscale = multiscale_from_axes(czyx());
        assert!(multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_valid_axes_length_zyx() {
        let multiscale = multiscale_from_axes(zyx());
        assert!(multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_valid_axes_length_yx() {
        let multiscale = multiscale_from_axes(yx());
        assert!(multiscale.is_axes_length_valid());
    }

    #[test]
    fn test_invalid_too_many_space_axes_types() {
        let multiscale = multiscale_from_axes(vec![x(), y(), z(), x()]);
        assert!(!multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_invalid_too_many_time_axes_types() {
        let multiscale = multiscale_from_axes(vec![x(), y(), z(), t(), t()]);
        assert!(!multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_invalid_too_many_channel_axes_types() {
        let multiscale = multiscale_from_axes(vec![x(), y(), z(), c(), c()]);
        assert!(!multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_invalid_too_many_custom_axes_types() {
        let multiscale = multiscale_from_axes(vec![x(), y(), z(), custom(), custom()]);
        assert!(!multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_tczyx() {
        let multiscale = multiscale_from_axes(tczyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_t_custom_zyx() {
        let multiscale = multiscale_from_axes(t_custom_zyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_czyx() {
        let multiscale = multiscale_from_axes(czyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_custom_zyx() {
        let multiscale = multiscale_from_axes(custom_zyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_zyx() {
        let multiscale = multiscale_from_axes(zyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_yx() {
        let multiscale = multiscale_from_axes(yx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_cyx() {
        let multiscale = multiscale_from_axes(cyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_custom_custom_yx() {
        let multiscale = multiscale_from_axes(custom_yx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_tzyx() {
        let multiscale = multiscale_from_axes(tzyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_types_tyx() {
        let multiscale = multiscale_from_axes(tyx());
        assert!(multiscale.are_axes_types_valid());
    }

    #[test]
    fn test_valid_axes_order_tczyx() {
        let multiscale = multiscale_from_axes(tczyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_t_custom_zyx() {
        let multiscale = multiscale_from_axes(t_custom_zyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_tzyx() {
        let multiscale = multiscale_from_axes(tzyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_tyx() {
        let multiscale = multiscale_from_axes(tzyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_tcyx() {
        let multiscale = multiscale_from_axes(tcyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_t_custom_yx() {
        let multiscale = multiscale_from_axes(t_custom_yx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_czyx() {
        let multiscale = multiscale_from_axes(czyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_custom_zyx() {
        let multiscale = multiscale_from_axes(custom_zyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_valid_axes_order_zyx() {
        let multiscale = multiscale_from_axes(zyx());
        assert!(multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_zyxtc() {
        let multiscale = multiscale_from_axes(vec![z(), y(), x(), t(), c()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_zyxct() {
        let multiscale = multiscale_from_axes(vec![z(), y(), x(), c(), t()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_zyxt_custom() {
        let multiscale = multiscale_from_axes(vec![z(), y(), x(), t(), custom()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_zyx_custom_t() {
        let multiscale = multiscale_from_axes(vec![z(), y(), x(), custom(), t()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_ctzyx() {
        let multiscale = multiscale_from_axes(vec![c(), t(), z(), y(), x()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_custom_t_zyx() {
        let multiscale = multiscale_from_axes(vec![custom(), t(), z(), y(), x()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_t_zyxc() {
        let multiscale = multiscale_from_axes(vec![t(), z(), y(), x(), c()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    #[test]
    fn test_invalid_axes_order_tzyx_custom() {
        let multiscale = multiscale_from_axes(vec![t(), z(), y(), x(), custom()]);
        assert!(!multiscale.is_axes_order_valid());
    }

    // todo: test coordinate_transformation validation
}
