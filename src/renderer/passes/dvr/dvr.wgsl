@group(0)
@binding(0)
var inputImage: texture_3d<u32>;

@group(0)
@binding(1)
var resultImage: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    z_slice: i32,
    z_max: f32
}

@group(0)
@binding(2)
var<uniform> z_slice: Uniforms;

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    t: f32,
    max_t: f32,
}

// supported builtins:
// - local_invocation_id    : vec3<u32>
// - local_invocation_index : u32
// - global_invocation_id   : vec3<u32>
// - workgroup_id           : vec3<u32>
// - num_workgroups         : vec3<u32>
@stage(compute)
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // todo:
    //  - generate ray
    //  - transform ray to camera's view space
    //  - set up volume bounds
    //  - step through volume until boundary is hit
    // -> required uniforms: camera.view, volume (3d texture + bounds)

    let pixel = bitcast<vec2<i32>>(global_id.xy);

    let raw_value = textureLoad(inputImage, vec3<i32>(pixel, z_slice.z_slice), 0).x;
    let value = vec3<f32>(f32(raw_value) / z_slice.z_max);
    textureStore(resultImage, pixel, vec4<f32>(value, 1.0));
}





struct LineState {
    point: vec3<i32>,
    err1: i32,
    err2: i32,
}

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
