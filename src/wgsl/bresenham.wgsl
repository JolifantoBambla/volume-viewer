@include(type_alias)

// https://gist.github.com/yamamushi/5823518
struct Line {
    // start point of the line
    start: int3,

    // x: dominant axis, y: secondary axis 1, z: secondary axis 2
    axis_indices: int3,

    // increment per axis
    inc: int3,

    // difference per axis
    d: int3,

    // max iterations until the end of line is reached
    max_iterations: i32,
}

struct LineState {
    p: int3,
    err1: i32,
    err2: i32,
}

fn create_bresenham3d(start: int3, end: int3) -> Line {
    let d1 = end - start;
    let lmn = abs(d);
    let d2 = d << 1;

    var inc = int3(1);
    for (var i = 0; i < 3; i = i + 1) {
        if d[i] < 0 {
            inc[i] = -1;
        }
    }
    // x: main axis, y: err1, z: err2
    var axis_indices = int3(2, 1, 0);
    if lmn[0] >= lmn[1] && lmn[0] >= lmn[2] {
        axis_indices = axis_indices.zyx;
    } else if lmn[1] >= lmn[0] && lmn[1] >= lmn[2] {
        axis_indices = axis_indices.yxz;
    }
    return Line(start, axis_indices, inc, d2, lms[axis_indices.x);
}

fn start_line(line: ptr<function, Line>) -> LineState {
    return LineSate {
        int3(*line.start),
        line.d[*line.axis_indices.y] - *line.max_iterations,
        line.d[*line.axis_indices.z] - *line.max_iterations
    }
}

fn next_voxel(state: ptr<function, LineState>, line: ptr<function, Line>) {
    if state.err1 > 0 {
        *state.p[*line.axis_indices.y] += *line.inc[*line.axis_indices.y];
        *state.err1 -= *line.d[*line.axis_indices.x];
    }
    if state.err2 > 0 {
        *state.p[*line.axis_indices.z] += *line.inc[*line.axis_indices.z];
        *state.err2 -= *line.d[*line.axis_indices.x];
    }
    *state.err1 += *line.d[*line.axis_indices.y];
    *state.err2 += *line.d[*line.axis_indices.z];
    *state.p[*line.axis_indices.x] += *line.inc[*line.axis_indices.x];
}

fn foo(start: int3, end: int3) {
    let line = create_bresenham3d(start, end);
    let line_state = start_line(&line);
    for (var i = 0; i < line.max_iterations; i += 1) {
        // do something with line_state.p

        next_voxel(&line_state, &line);
    }
}
