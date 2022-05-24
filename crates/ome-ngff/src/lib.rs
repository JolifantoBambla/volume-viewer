//! OME-NGFF Metadata
//! https://ngff.openmicroscopy.org/0.4/
//! https://ngff.openmicroscopy.org/latest/#metadata

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

pub mod axes;
pub mod coordinate_transformations;
pub mod image_labels;
pub mod multi_scales;
pub mod omero; // todo: test this, check out spec
pub mod plate;
pub mod validation;
pub mod well;

pub use axes::{
    Axis,
    ChannelAxis,
    CustomAxis,
    SpaceAxis,
    SpaceUnit,
    TimeAxis,
    TimeUnit,
};
pub use coordinate_transformations::{
    CoordinateTransformation,
    Identity,
    Scale,
    Translation,
};
pub use image_labels::{
    ImageLabel,
    Property,
    Source,
};
pub use multi_scales::{
    Dataset,
    MultiScale,
};
pub use omero::{
    Channel,
    Omero,
    RDefs,
    Window,
};
pub use plate::{
    Plate,
};
pub use validation::{

};
pub use well::{
    Well,
};

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    #[serde(rename = "multiscales")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multi_scales: Option<Vec<multi_scales::MultiScale>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub omero: Option<Omero>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub labels: Option<Vec<String>>,

    // if present, represents image segmentation
    // the two "dataset" series must have same number of entries (wtf are the two dataset entries?)
    #[serde(rename = "image-label")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_label: Option<ImageLabel>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub plate: Option<Plate>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub well: Option<Well>,
}
