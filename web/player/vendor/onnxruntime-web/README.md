# onnxruntime-web (vendored)

Pinned copy of the **pure-WASM** build of [onnxruntime-web] **1.27.0**, taken
straight from the npm tarball (`npm pack onnxruntime-web@1.27.0`, files from
`package/dist/`). It is vendored rather than pulled from a CDN so the player at
`web/player/` keeps working offline once loaded and has no third-party
references at runtime.

| file | size | what it is |
| --- | --- | --- |
| `ort.wasm.bundle.min.mjs` | 73 KB | the ES-module API (`InferenceSession`, `Tensor`, `env`) |
| `ort-wasm-simd-threaded.wasm` | 13.5 MB | the runtime itself (SIMD; runs single-threaded here) |

The `bundle` variant is the one vendored on purpose: it inlines the Emscripten
glue (`ort-wasm-simd-threaded.mjs`) instead of importing it at runtime, so the
only path that has to resolve correctly is the `.wasm` — one less thing to get
wrong inside a module worker.

The WebGPU / WebGL / training / asyncify builds are deliberately not vendored:
the net is a 3.3 M-parameter MLP evaluated one position at a time, where WASM is
already the fastest option and the WebGPU wasm blob is twice the size.

The page sets `ort.env.wasm.numThreads = 1`: SharedArrayBuffer needs
cross-origin isolation (COOP/COEP), which GitHub Pages does not send, and a
batch-of-one MLP gains nothing from threads anyway.

To update: bump the version above, re-run `npm pack onnxruntime-web@<version>`,
copy the two files, then re-run `node web/player/test/parity.test.mjs` (which
checks this runtime against the torch reference) and
`node web/player/test/selfplay.test.mjs`.

onnxruntime-web is MIT licensed, Copyright (c) Microsoft Corporation.

[onnxruntime]: https://github.com/microsoft/onnxruntime
[onnxruntime-web]: https://www.npmjs.com/package/onnxruntime-web
