@include(type_alias)

struct Transform {
    // Maps a point in the object's space to a common world space.
    object_to_world: float4x4,

    // Maps a point in a common world space to the object's space.
    // It is the inverse of `object_to_world` (WGSL doesn't have a built-in function to compute the inverse of a matrix).
    world_to_object: float4x4,
}