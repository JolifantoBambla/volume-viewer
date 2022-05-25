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
