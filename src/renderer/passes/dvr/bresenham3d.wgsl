// https://gist.github.com/yamamushi/5823518
struct Line {
    // start point of the line
    start: vec3<i32>,

    // x: dominant axis, y: secondary axis 1, z: secondary axis 2
    axis_indices: vec3<i32>,

    // increment per axis
    inc: vec3<i32>,

    // difference per axis
    d: vec3<i32>,

    // max iterations until the end of line is reached
    max_iterations: i32,
}

struct LineState {
    point: vec3<i32>,
    err1: i32,
    err2: i32,
}

fn create_bresenham3d(start: vec<i32>, end: vec<i32>) -> Line {
    let d1: vec<i32> = end - start;
    let lmn: vec<i32> = abs(d);
    let d2: vec<i32> = d << 1;

    var inc: vec<i32> = vec<i32>(1);
    for (var i: i32 = 0; i < 3; i = i + 1) {
        if d[i] < 0 {
            inc[i] = -1;
        }
    }
    // x: main axis, y: err1, z: err2
    var axis_indices = vec3<i32>(2, 1, 0);
    if lmn[0] >= lmn[1] && lmn[0] >= lmn[2] {
        axis_indices = axis_indices.zyx; //vec3<i32>(0, 1, 2);
    } else if lmn[1] >= lmn[0] && lmn[1] >= lmn[2] {
        axis_indices = axis_indices.yxz; //vec3<i32>(1, 0, 2);
    }
    return Line(start, axis_indices, inc, d2, lms[axis_indices.x);
}

fn start_line(line: ptr<function, Line>) -> LineState {
    return LineSate {
        vec3<i32>(*line.start),
        line.d[*line.axis_indices.y] - *line.max_iterations,
        line.d[*line.axis_indices.z] - *line.max_iterations
    }
}

fn next_voxel(state: ptr<function, LineState>, line: ptr<function, Line>) {
    if state.err1 > 0 {
        *state.point[*line.axis_indices.y] += *line.inc[*line.axis_indices.y];
        *state.err1 -= *line.d[*line.axis_indices.x];
    }
    if state.err2 > 0 {
        *state.point[*line.axis_indices.z] += *line.inc[*line.axis_indices.z];
        *state.err2 -= *line.d[*line.axis_indices.x];
    }
    *state.err1 += *line.d[*line.axis_indices.y];
    *state.err2 += *line.d[*line.axis_indices.z];
    *state.point[*line.axis_indices.x] += *line.inc[*line.axis_indices.x];
}

fn foo(start: vec3<i32>, end: vec3<i32>) {
    let line = create_bresenham3d(start, end);
    let line_state = start_line(&line);
    for (var i = 0; i < line.max_iterations; i += 1) {
        // do something with line_state.point

        next_voxel(&line_state, &line);
    }
}
