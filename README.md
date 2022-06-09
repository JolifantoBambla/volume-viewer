# volume-viewer
A web-based volume viewer for large-scale multichannel volume data

**Currently, this is more of a playground.**

## Install build dependencies
* [Install Rust](https://www.rust-lang.org/tools/install)
* Install `wasm-pack`: `cargo install wasm-pack`
* From the project root run: `RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web`
* Start a webserver in the project root (e.g. `http-server` or `python3 -m http.server`)
* Using a [browser that supports WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), navigate to `localhost:<whatever-port-you-serve-the-page-on>` (e.g. on Linux: Chromium with flags `--enable-vulkan --enable-unsafe-webgpu`)
