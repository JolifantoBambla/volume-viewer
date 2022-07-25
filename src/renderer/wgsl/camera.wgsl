@include(constant)
@include(ray)
@include(transform)
@include(type_alias)

type CameraType = u32;
let PERSPECTIVE = 0u;
let ORTHOGRAPHIC = 1u;

struct Camera {
    // Maps points from/to the camera's space to/from a common world space.
    transform: Transform,

    // Projects a point in the camera's object space to the camera's image plane.
    // This field is only needed for surface rendering.
    projection: float4x4,

    // The inverse of the `projection` matrix.
    // (WGSL doesn't have a built-in function to compute the inverse of a matrix).
    inverse_projection: float4x4,

    // The type of this camera:
    //  1:  Orthographic
    //  else: Perspective
    // Note that this field has a size of 16 bytes to avoid alignment issues.
    @size(16) camera_type: CameraType,
}

fn raster_to_screen(pixel: float2, resolution: float2) -> float2 {
    let aspect_ratio = resolution.x / resolution.y;
    var screen_min = float2(-aspect_ratio, -1.);
    var screen_max = float2( aspect_ratio,  1.);
    if aspect_ratio < 1. {
        var screen_min = float2(-1., -1. / aspect_ratio);
        var screen_max = float2( 1.,  1. / aspect_ratio);
    }
    return float2(
        pixel.x * ((screen_max.x - screen_min.x) / resolution.x) - screen_max.x,
        pixel.y * ((screen_min.y - screen_max.y) / resolution.y) + screen_max.y
    );
}

// Generates a ray in a common "world" space from a `Camera` instance.
fn generate_camera_ray(camera: Camera, pixel: float2, resolution: float2) -> Ray {
    let offset = 0.5;
    let camera_point = (camera.inverse_projection * float4(raster_to_screen(pixel + offset, resolution), 0., 1.)).xyz;

    var origin = float3();
    var direction = float3();

    if camera.camera_type == ORTHOGRAPHIC {
        origin = camera_point;
        direction = float3(0., 0., -1.);
    } else {
        direction = normalize(camera_point);
    }

    return transform_ray(
        Ray (origin, direction, positive_infinity()),
        camera.transform.object_to_world
    );
}
