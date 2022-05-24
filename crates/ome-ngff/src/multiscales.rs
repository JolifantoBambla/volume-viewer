use serde::{Serialize, Deserialize};

use crate::axes;
use crate::coordinate_transformations;

#[derive(Serialize, Deserialize)]
pub struct Dataset {
    pub path: String,

    #[serde(rename = "coordinateTransformations")]
    pub coordinate_transformations: Vec<coordinate_transformations::CoordinateTransformation>,
}

#[derive(Serialize, Deserialize)]
pub struct MultiScale {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub downscaling_type: Option<String>,

    pub axes: Vec<axes::Axis>,

    // ordered by largest (i.e. highest resolution) to smallest.
    pub datasets: Vec<Dataset>,

    // are applied after `coordinate_transformations` in `datasets`
    #[serde(rename = "coordinateTransformations")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_transformations: Option<Vec<coordinate_transformations::CoordinateTransformation>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}