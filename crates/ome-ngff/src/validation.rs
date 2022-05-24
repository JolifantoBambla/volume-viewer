// todo:
//  - "multiscales.axes": length must be 2-5
//  - "multiscales.axes": 2-3 entries must be "type:space", one may be "type:time", one may be "type:channel", one may be a custom type
//  - "multiscales.axes": time axis must be first, followed by channel or custom axis, followed by space axes (space axes should be ordered ZYX)
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
//  - "image-label": if present, "multiscales" must also be present
//  - "image-label": if present, the two "dataset" entries (in "multiscales" or where?) must have same number of entries
//  - "image-label.colors": "label-value"s should be unique
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
//  - "well.images[*].acquisition": if "plate.acquisitions" has more than one entry, "acquisition" must not be None
