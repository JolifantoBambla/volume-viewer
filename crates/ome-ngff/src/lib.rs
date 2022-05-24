//! OME-NGFF Metadata
//! https://ngff.openmicroscopy.org/0.4/
//! https://ngff.openmicroscopy.org/latest/#metadata

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub mod axes;
pub mod coordinate_transformations;
pub mod multiscales;
pub mod omero;
pub mod validation;

pub use axes::*;
pub use coordinate_transformations::*;
pub use multiscales::*;
pub use omero::*;
pub use validation::*;

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    #[serde(rename = "multiscales")]
    pub multi_scales: Vec<multiscales::MultiScale>,
}

#[derive(Serialize, Deserialize)]
pub struct Example {
    pub field1: HashMap<u32, String>,
    pub field2: Vec<Vec<f32>>,
    pub field3: [f32; 4],
    pub axes: Vec<axes::Axis>,

    #[serde(rename = "coordinateTransformations")]
    pub coordinate_transformations: Vec<coordinate_transformations::CoordinateTransformation>,
}

pub fn foo() {
    log::info!("logging from ome-ngff");
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
