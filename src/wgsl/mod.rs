use wgsl_preprocessor::WGSLPreprocessor;

pub fn create_wgsl_preprocessor() -> WGSLPreprocessor {
    let mut wgsl_preprocessor = WGSLPreprocessor::default();

    wgsl_preprocessor
        .include("aabb", include_str!("aabb.wgsl"))
        .include("bresenham", include_str!("bresenham.wgsl"))
        .include("camera", include_str!("camera.wgsl"))
        .include("cache_update_meta", include_str!("cache_update_meta.wgsl"))
        .include("channel_settings", include_str!("channel_settings.wgsl"))
        .include("constant", include_str!("constant.wgsl"))
        .include("dispatch_indirect", include_str!("dispatch_indirect.wgsl"))
        .include("global_settings", include_str!("gobal_settings.wgsl"))
        .include("gpu_list", include_str!("gpu_list.wgsl"))
        .include("grid_traversal", include_str!("grid_traversal.wgsl"))
        .include("grid_leap", include_str!("grid_leap.wgsl"))
        .include("lighting", include_str!("lighting.wgsl"))
        .include(
            "multichannel_octree_util",
            include_str!("multichannel_octree_util.wgsl"),
        )
        .include("octree_node", include_str!("octree_node.wgsl"))
        .include("octree_node_util", include_str!("octree_node_util.wgsl"))
        .include(
            "octree_node_write_util",
            include_str!("octree_node_write_util.wgsl"),
        )
        .include("output_modes", include_str!("output_modes.wgsl"))
        .include(
            "page_directory_util",
            include_str!("page_directory_util.wgsl"),
        )
        .include(
            "page_directory_meta_util",
            include_str!("page_directory_meta_util.wgsl"),
        )
        .include("page_table", include_str!("page_table.wgsl"))
        .include("page_table_util", include_str!("page_table_util.wgsl"))
        .include("ray", include_str!("ray.wgsl"))
        .include("sphere", include_str!("sphere.wgsl"))
        .include("timestamp", include_str!("timestamp.wgsl"))
        .include("transform", include_str!("transform.wgsl"))
        .include("type_alias", include_str!("type_alias.wgsl"))
        .include("util", include_str!("util.wgsl"))
        .include("volume_accessor", include_str!("volume_accessor.wgsl"))
        .include(
            "volume_subdivision",
            include_str!("volume_subdivision.wgsl"),
        )
        .include(
            "volume_subdivision_util",
            include_str!("volume_subdivision_util.wgsl"),
        )
        .include("volume_util", include_str!("volume_util.wgsl"));

    wgsl_preprocessor
}
