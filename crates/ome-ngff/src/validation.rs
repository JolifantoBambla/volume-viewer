// todo: multiscales
//  - "multiscales.datasets": all entries must have the same number of dimensions and these have to be less than or exactly 5
//  - "multiscales.datasets[*].coordinateTransformations": must only be of type translation or scale
//  - "multiscales.datasets[*].coordinateTransformations": must contain exactly one scale
//  - "multiscales.datasets[*].coordinateTransformations": may contain exactly one translation
//  - "multiscales.datasets[*].coordinateTransformations": translation must come after scale
//  - "multiscales.datasets[*].coordinateTransformations": length of translation or scale array must be same as "multiscales.axes"
//  - "multiscales.coordinateTransformations": must only be of type translation or scale
//  - "multiscales.coordinateTransformations": must contain exactly one scale
//  - "multiscales.coordinateTransformations": may contain exactly one translation
//  - "multiscales.coordinateTransformations": translation must come after scale
//  - "multiscales.coordinateTransformations": length of translation or scale array must be same as "multiscales.axes"

use log::warn;
use crate::axes::{
    Axis,
    ChannelAxis,
    CustomAxis,
    SpaceAxis,
    TimeAxis,
};

macro_rules! warn_on_false {
    ( $condition:expr, $message:expr $(, $format_arg:expr)* ) => {
        if $condition {
            true
        } else {
            warn!($message $(,$format_arg)*);
            false
        }
    }
}

pub fn validate_multi_scale_axes_length(axes: &Vec<Axis>) -> bool {
    warn_on_false!(
        axes.len() >= 2 && axes.len() <= 5,
        "The spec states: The length of \"axes\" must be between 2 and 5. Got: {}",
        axes.len()
    )
}

pub fn validate_multi_scale_axes_types(axes: &Vec<Axis>) -> bool {
    let space_count = Axis::get_space_axes(axes).len();
    let time_count = Axis::get_time_axes(axes).len();
    let channel_count = Axis::get_channel_axes(axes).len();
    let custom_count = Axis::get_custom_axes(axes).len();
    warn_on_false!(
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
        custom_count
    )
}

pub fn validate_multi_scale_axes_order(axes: &Vec<Axis>) -> bool {
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
    warn_on_false!(
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
        space_indices
    )
}

pub fn validate_multi_scale_axes(axes: &Vec<Axis>) -> bool {
    validate_multi_scale_axes_length(axes) &&
        validate_multi_scale_axes_types(axes) &&
        validate_multi_scale_axes_order(axes)
}

// todo: image-label
//  - "image-label": if present, "multiscales" must also be present
//  - "image-label": if present, the two "dataset" entries (in "multiscales" or where?) must have same number of entries
//  - "image-label.colors": "label-value"s should be unique

// todo: plate
//  - "plate.columns": each column in physical plate must be defined, even if no wells in the columns are defined
//  - "plate.columns[*].name": must contain only alphanumeric characters
//  - "plate.columns[*].name": must be case-sensitive
//  - "plate.columns[*].name": must not be a duplicate of any other name in "plate.columns"
//  - "plate.rows": each row in physical plate must be defined, even if no wells in the rows are defined
//  - "plate.rows[*].name": must contain only alphanumeric characters
//  - "plate.rows[*].name": must be case-sensitive
//  - "plate.rows[*].name": must not be a duplicate of any other name in "plate.columns"
//  - "plate.wells[*].path": must consist of a "plate.rows[*].name", a file separator (/), and a "plate.columns[*].name"
//  - "plate.wells[*].path": must not not contain additional leading or trailing directories
//  - "plate.wells[*].rowIndex": must be an index into "plate.rows[*]"
//  - "plate.wells[*].columnIndex": must be an index into "plate.columns[*]"
//  - "plate.wells": "path" and "rowIndex"+"columnIndex" pair must refer to same row/column pair

// todo: well
//  - "well.images[*].acquisition": if "plate.acquisitions" has more than one entry, "acquisition" must not be None
