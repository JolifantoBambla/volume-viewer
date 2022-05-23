use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// https://ngff.openmicroscopy.org/0.4/

pub mod axes;
pub mod coordinate_transformations;
pub mod multiscales;
pub mod omero;

pub use axes::*;
pub use coordinate_transformations::*;
pub use multiscales::*;
pub use omero::*;

// https://ngff.openmicroscopy.org/latest/#metadata

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
    pub axis: axes::Axis,
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
