//! OME-NGFF Metadata
//! https://ngff.openmicroscopy.org/0.4/
//! https://ngff.openmicroscopy.org/latest/#metadata

pub mod axis;
pub mod coordinate_transformations;
pub mod image_labels;
pub mod metadata;
pub mod multiscale;
pub mod omero; // todo: test this, check out spec
pub mod plate;
pub mod util;
pub mod well;
