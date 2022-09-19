mod shared_constants;

use wgsl_preprocessor::WGSLPreprocessor;

pub fn create_wgsl_preprocessor() -> WGSLPreprocessor {
    let mut wgsl_preprocessor = WGSLPreprocessor::default();

    wgsl_preprocessor
        .include("aabb", include_str!("aabb.wgsl"))
        .include("bresenham", include_str!("bresenham.wgsl"))
        .include("camera", include_str!("camera.wgsl"))
        .include("constant", include_str!("constant.wgsl"))
        .include("gpu_list", include_str!("gpu_list.wgsl"))
        .include("grid_traversal", include_str!("grid_traversal.wgsl"))
        .include("page_table", include_str!("page_table.wgsl"))
        .include("ray", include_str!("ray.wgsl"))
        .include("sphere", include_str!("sphere.wgsl"))
        .include("transform", include_str!("transform.wgsl"))
        .include("type_alias", include_str!("type_alias.wgsl"))
        .include("util", include_str!("util.wgsl"))
        .include("volume_util", include_str!("volume_util.wgsl"));

    wgsl_preprocessor
}
