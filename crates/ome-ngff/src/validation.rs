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
