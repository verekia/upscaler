# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Upscaler is a single-page, client-only AI image upscaler: Real-ESRGAN (the official checkpoints that
`realesrgan-ncnn-vulkan` ships, which manablade's `bun upscale` runs natively) executed in the browser with
ONNX Runtime Web, on WebGPU or WebAssembly. Nothing is uploaded; the page only downloads the weights.

## Development Commands

```bash
bun install
bun dev            # assets, then portless upscaler next dev (Next 16, Turbopack)
bun run build      # assets, then the static export (output: 'export')
bun run assets     # scripts/assets.ts alone: public/models and public/ort
bun run all        # format:check + lint + typecheck + warden + test — run before considering work done
bun run test       # bun test
```

## Architecture

Next.js Pages Router as a static shell: `pages/index.tsx` dynamically imports `src/App.tsx` with
`ssr: false`. Everything heavy happens in one module worker.

- **`scripts/assets.ts`** — downloads the pinned `.pth` checkpoints (`CHECKPOINTS` in `src/models.ts`) into
  `~/.cache/upscaler`, checks the sha256, and writes `public/models/<checkpoint>.safetensors` (4D conv kernels
  fp16, biases and PReLU slopes fp32). Skips files whose metadata already matches; bump `CONVERTER` to force.
  Also copies ONNX Runtime's `ort.webgpu.min.mjs` / `ort.wasm.min.mjs` and their glue + wasm into
  `public/ort`, and writes the `NOTICES` beside what they cover: Real-ESRGAN's BSD-3 license as
  `public/models/LICENSE.txt`, ONNX Runtime's license and third-party notices (absent from its npm package,
  fetched from its release tag) into `public/ort`. Bumping `onnxruntime-web` fails here until `ORT_VERSION`
  and those notices' sha256 match it. All of `public/` is gitignored and rebuilt in Docker.
- **`pages/licenses.tsx`** — the Licenses page (linked from the header and the sidebar's credit), with the
  license texts read from `public/` at build time by `getStaticProps`.
- **`scripts/pth.ts`** — a PyTorch checkpoint reader with no PyTorch: zip central directory + the small
  pickle subset `torch.save` emits for a dict of tensors.
- **`src/onnx.ts`** — a minimal ONNX protobuf writer (`Graph.op`, `Graph.initializer`, `Graph.encode`).
- **`src/graph.ts`** — `buildModel` writes RRDBNet (x4plus models) or SRVGGNetCompact (v3 models) from a
  state dict, detected from its keys, in fp16 or fp32, with dynamic H×W. `checkpointWeights` blends two
  checkpoints by the denoise strength (Real-ESRGAN's DNI).
- **`src/upscale.ts`** — the pipeline, pure: equal tiles with overlap (`src/tiles.ts`: reflect-101 borders
  like ncnn, TTA transforms), the alpha pass, and one row of tiles at a time into `RowResampler`
  (`src/resize.ts`), which streams rows so memory follows the output, never the 4× intermediate.
- **`src/upscale.worker.ts`** — loads ONNX Runtime from `/ort/` at run time, fetches and caches checkpoints,
  caches one session per model + precision + denoise, runs the pipeline, encodes (`src/png.ts` for PNG,
  OffscreenCanvas for JPEG/WebP). Messages are typed in `src/worker-protocol.ts`.
- **`src/engine.ts`** — the main thread's worker handle: one worker per backend, `auto` tries WebGPU then
  wasm; switching backend replaces the worker.
- **`src/options.ts`** — the settings, their defaults (manablade's: anime model, 2×, Lanczos3), persistence,
  output size and limits.
- **`src/App.tsx`**, **`src/components/`** — the UI: settings sidebar, the split viewer (`Viewer.tsx`) with a
  live tile canvas, drop/paste/open.

## Conventions & gotchas

- **Warden** (`@verekia/warden`, run by `bun run all`) enforces shared config/versions across the user's
  repos. Keep `next`/`react`/`typescript`/`oxfmt`/`oxlint`/etc. at their pinned versions and the
  `format`/`lint`/`warden` scripts intact. **Avoid adding npm dependencies**: `onnxruntime-web` is the only
  runtime one.
- **ONNX Runtime is not bundled.** The worker `import()`s `/ort/ort.<backend>.min.mjs` (with
  `webpackIgnore`/`turbopackIgnore`) so its pthread workers load its own glue file instead of re-running the
  app's worker bundle. Its types come from `import type * as Ort from 'onnxruntime-web'`.
- **Cross-origin isolation** (COOP/COEP) gives the wasm backend its threads: `next.config.js` `headers` for
  `next dev`, `nginx.conf` for production. In `nginx.conf`, locations only use `expires`: an `add_header`
  in a location silently drops the server-level ones, COOP/COEP included.
- **Fidelity to ncnn**: borders are mirrored (reflect-101) as in `realesrgan-ncnn-vulkan`'s preprocessing,
  and rounding is Uint8ClampedArray's (half to even, like numpy's `round`). Checked at 45-55 dB PSNR against
  ncnn 20220424 for all three architectures. Don't switch to replicate padding: the borders drift.
- **One tile shape per image**: `tileSize` makes all tiles equal and the last one moves back inside the
  image, so ONNX Runtime compiles its shaders for one shape. TTA needs square tiles (transposes).
- **Worker bundling**: Turbopack bundles `new Worker(new URL('./upscale.worker.ts', import.meta.url))` but also
  copies the raw `.ts` into `out/_next/static/media/`, which is why `tsconfig` excludes `out`. Import message
  types from `worker-protocol.ts`, never from the worker module.
- **TypeScript 6** defaults `types` to `[]`, hence `"types": ["bun"]` for the scripts and tests.
- **`Bun.write(file, otherFile)`** writes nothing when the target's `.size` was read while it didn't exist:
  `scripts/assets.ts` copies with `node:fs`.
- **Tests** (`bun test`) cover the fp16 conversion, tiling/TTA, the streaming resampler, the PNG encoder, the
  pipeline with a stand-in model, and both graphs against plain-JS reference implementations run through
  ONNX Runtime's wasm backend.
- **Formatting/lint**: oxfmt (no semicolons, single quotes, 120 cols, trailing commas) + oxlint.
