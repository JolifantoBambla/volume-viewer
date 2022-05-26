use serde::{Serialize, Deserialize};

use crate::multiscale::Multiscale;
use crate::omero::Omero;
use crate::image_label::ImageLabel;
use crate::plate::Plate;
use crate::well::Well;

// https://ngff.openmicroscopy.org/latest/#metadata

#[derive(Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiscales: Option<Vec<Multiscale>>,

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

// todo: image-label validation
//  - "image-label": if present, "multiscales" must also be present
//  - "image-label": if present, the two "dataset" entries (in "multiscales" or where?) must have same number of entries

// todo: well validation
//  - "well.images[*].acquisition": if "plate.acquisitions" has more than one entry, "acquisition" must not be None
