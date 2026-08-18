# onnxruntime-web (vendored)

Pinned copy of [onnxruntime-web] **1.27.0**, taken straight from the npm tarball
(`npm pack onnxruntime-web@1.27.0`, files from `package/dist/`). It is vendored
rather than pulled from a CDN so the player at `web/player/` keeps working
offline once loaded and has no third-party references at runtime.

**Two builds are vendored and exactly one is downloaded per visitor.** The
worker feature-detects (`js/worker.js` → `js/net.js`, `BACKENDS`) and asks for
the first one that works; a browser without WebGPU never requests the WebGPU
runtime at all.

| file | size | gzip | what it is |
| --- | --- | --- | --- |
| `ort.wasm.bundle.min.mjs` | 73 KB | 24 KB | the ES-module API, pure-WASM build |
| `ort-wasm-simd-threaded.wasm` | 13.5 MB | 3.44 MB | the CPU runtime (SIMD; runs single-threaded here) |
| `ort.jspi.bundle.min.mjs` | 110 KB | 37 KB | the same API, WebGPU build |
| `ort-wasm-simd-threaded.jspi.wasm` | 15.0 MB | 3.67 MB | CPU kernels **plus** the native WebGPU execution provider |

The `bundle` variants are the ones vendored on purpose: they inline the
Emscripten glue instead of importing it at runtime, so the only path that has to
resolve correctly is the `.wasm` — one less thing to get wrong inside a module
worker.

## Why the JSPI build, and not the other two WebGPU builds

onnxruntime 1.27 ships the WebGPU execution provider in three flavours, which
differ only in how the WASM call is suspended while the GPU works:

| build | wasm | gzip | needs |
| --- | --- | --- | --- |
| `ort.jspi.*` | `…jspi.wasm` | 3.67 MB | `WebAssembly.Suspending` (JSPI) |
| `ort.webgpu.*` | `…asyncify.wasm` | 5.95 MB | nothing beyond WebGPU |
| `ort.*` (JSEP) | `…jsep.wasm` | 6.31 MB | nothing beyond WebGPU |

JSPI costs **+0.23 MB gzipped over the pure-WASM build** and was measured
fastest of the three (16.5 k vs 11.2 k vs 9.5 k positions/s at batch 64 on an
M3 Pro). The asyncify build would let Safari and Firefox onto the GPU too, but
at +2.5 MB gzipped for every visitor who takes that path — not a trade worth
making for a fallback that already runs at a perfectly good 3 k positions/s.
When those browsers ship JSPI, they get the GPU path with no change here.

The WebGL and training builds are not vendored at all.

## Threads

Both builds set `ort.env.wasm.numThreads = 1`: SharedArrayBuffer needs
cross-origin isolation (COOP/COEP), which GitHub Pages does not send.

## Updating

Bump the version above, re-run `npm pack onnxruntime-web@<version>`, copy the
four files, then re-run the gates:

```
node web/player/test/parity.test.mjs    # WASM runtime vs the torch reference
node web/player/test/webgpu.test.mjs    # WebGPU runtime vs the same reference
node web/player/test/margin.test.mjs    # output detection + batched read-out
node web/player/test/selfplay.test.mjs  # whole games through the whole stack
```

onnxruntime-web is MIT licensed, Copyright (c) Microsoft Corporation.

[onnxruntime]: https://github.com/microsoft/onnxruntime
[onnxruntime-web]: https://www.npmjs.com/package/onnxruntime-web
