# volume-viewer
A web-based volume viewer for large-scale multichannel volume data

## Install build dependencies
* [Install Rust](https://www.rust-lang.org/tools/install)
* Install `wasm-pack`: `cargo install wasm-pack`
* From the project root run: `RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web`
* Start a webserver in the project root (e.g. `http-server` or `python3 -m http.server`)
* Using a [browser that supports WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), navigate to `localhost:<whatever-port-you-serve-the-page-on>` (e.g. on Linux: Chromium with flags `--enable-vulkan --enable-unsafe-webgpu`)

**NOTE: this project currently targets Chromium-based browsers (Chromium, Chrome, Edge).**

## Build

```
RUSTFLAGS=--cfg\=web_sys_unstable_apis\ -C\ target-feature\=+atomics\,+bulk-memory\,+mutable-globals \
RUSTUP_TOOLCHAIN=nightly \
wasm-pack build --target web -- . -Z build-std=panic_abort,std
```

### Required Rustflags
* `--cfg=web_sys_unstable_apis`: enables unstable APIs like WebGPU for the `web_sys` crate
* `-C target-feature=+atomics,+bulk-memory,+mutable-globals`: enables atomics & shared memory for WASM. This is required for multithreading.

### Required Unstable Flags
* `build-std=panic_abort,std`: rebuild std with the features (atomics, etc.) enabled by `RUSTFLAGS`

## Run
This project requires cross-origin isolation headers to be set.
Use the `server.py` script provided by this project to serve the built package.

