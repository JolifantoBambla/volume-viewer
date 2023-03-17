@group(0) @binding(0) var screen_sampler : sampler;
@group(0) @binding(1) var screen_texture : texture_2d<f32>;
@group(0) @binding(2) var<uniform> background_color: vec4<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) fragUV : vec2<f32>,
};

@vertex
fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  var pos = array<vec2<f32>, 6>(
      vec2<f32>( 1.0,  1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0,  1.0));

  var uv = array<vec2<f32>, 6>(
      vec2<f32>(1.0, 0.0),
      vec2<f32>(1.0, 1.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(1.0, 0.0),
      vec2<f32>(0.0, 1.0),
      vec2<f32>(0.0, 0.0));

  var output : VertexOutput;
  output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = uv[VertexIndex];
  return output;
}

@fragment
fn frag_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
  let dvr_color = textureSample(screen_texture, screen_sampler, fragUV);
  if (any(dvr_color > vec4<f32>())) {
    return blend(dvr_color, background_color);
  } else {
    return background_color;
  }
}

fn blend(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.rgb * a.a + b.rgb * b.a * (1. - a.a),
        a.a + b.a * (1. - a.a)
    );
}