# volume-viewer
A web-based volume viewer for large-scale multichannel volume data

## Install build dependencies
* [Install Rust](https://www.rust-lang.org/tools/install)
* [Install wasm-pack](https://rustwasm.github.io/wasm-pack/installer)

### Additional Dependencies for the Demo
* From the project root run: `npm install`
* Install an HTTP server that allows you to enable CORS-headers (e.g., [http-server](https://www.npmjs.com/package/http-server)) to serve your data.
* Install a [browser that supports WebGPU](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status), navigate to `localhost:<whatever-port-you-serve-the-page-on>` (e.g. on Linux: Chromium with flags `--enable-vulkan --enable-unsafe-webgpu`)

**NOTE: this project currently targets Chromium-based browsers (Chromium, Chrome, Edge).**

## Build

From the project root run:
```
wasm-pack build --target web
```

### Required Rustflags
These are set from `.cargo/config.toml`.
* `--cfg=web_sys_unstable_apis`: enables unstable APIs like WebGPU for the `web_sys` crate
* `-C target-feature=+atomics,+bulk-memory,+mutable-globals`: enables atomics & shared memory for WASM. This is required for multithreading.

### Required Unstable Flags
These are set from `.cargo/config.toml`.
The nightly toolchain used to build this project is defined in `rust-toolchain.toml`.
* `build-std=panic_abort,std`: rebuild std with the features (atomics, etc.) enabled by `RUSTFLAGS`

### Timestamp Queries
This crate provides a `timestamp-query` feature which enables timing GPU passes via WebGPU's timestamp query.
To use this feature, Google Chrome has to be run with the following flag enabled:
```
--disable-dawn-features=disallow_unsafe_apis
```
Note that this flag is currently not supported on Linux (and probably MacOS).

## Demo
Check out [the wiki](https://github.com/JolifantoBambla/volume-viewer/wiki/Demo) for more information on how to try out the demo.
