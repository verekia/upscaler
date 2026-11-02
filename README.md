# Upscaler

**AI image upscaling with Real-ESRGAN, entirely in your browser.** Drop an image, pick a model, and get it
back at 2×, 4× or any size, sharper than any resampling can make it. Nothing is uploaded: the network runs
on your GPU with WebGPU (or on the CPU with WebAssembly), from weights the page downloads once.

```bash
bun install
bun dev
```

## Models

The official Real-ESRGAN checkpoints, the same networks `realesrgan-ncnn-vulkan` ships:

| Model       | Checkpoint                                    | Best for                                                 |
| ----------- | --------------------------------------------- | -------------------------------------------------------- |
| Anime       | `realesrgan-x4plus-anime` · RRDBNet, 6 blocks | Anime, illustrations, game art. The default.             |
| Photo       | `realesrgan-x4plus` · RRDBNet, 23 blocks      | Photos and realistic renders. The slowest, ~3× Anime.    |
| Anime video | `realesr-animevideov3` · SRVGGNet, 16 convs   | Video frames: ~9× faster than Anime, a little softer.    |
| General     | `realesr-general-x4v3` · SRVGGNet, 32 convs   | Photos and mixed content, fast, with a denoise strength. |

Every model upscales 4×. ncnn's `realesr-animevideov3-x2` and `-x3` are that same x4 network followed by a
bicubic downscale; here any output size is the 4× result resampled.

## Settings

- **Model** and, for General, the **denoise strength**: Real-ESRGAN's network interpolation between
  `realesr-general-x4v3` (1, removes noise) and its weak-denoise twin (0, keeps grain).
- **Output size** by scale, width or height. The 4× result is resampled to it with Lanczos3 (as sharp does
  in manablade's `bun upscale`), Mitchell (softer, no ringing) or Catmull-Rom.
- **Format**: PNG (lossless, any size, exact pixels), JPEG or WebP with a quality.
- **Test-time augmentation**: the average of the model over the 8 flips and transposes of each tile. A touch
  cleaner, 8× slower (ncnn's `-x`).
- **Transparency**, for images with an alpha channel: through the model like the colors (Real-ESRGAN's
  default, 2× the work) or resampled.
- **Advanced**: backend (WebGPU or CPU), precision (FP16 needs WebGPU's `shader-f16`), tile size, tile overlap
  and the resampling filter.

Settings persist in `localStorage`. Drop an image anywhere, paste it, or open it; the viewer splits original
and result (drag the divider, wheel to zoom, drag to pan, `F` fits, `1` shows actual size).

## How it works

1. **Weights.** `scripts/assets.ts` (run by `dev` and `build`) downloads the pinned `.pth` checkpoints into
   `~/.cache/upscaler`, checks their sha256, reads them without PyTorch (a zip of a pickle) and writes
   `public/models/*.safetensors`, convolutions in fp16 like the ncnn release. It also copies ONNX Runtime
   Web's ESM build and wasm into `public/ort`, and puts each one's license beside it (pinned by sha256 too).
2. **Graph.** A worker builds the network as an ONNX graph from the state dict (`src/graph.ts`, written by a
   ~150-line protobuf writer in `src/onnx.ts`), in fp16 or fp32, with the general model's two checkpoints
   blended by the denoise strength, and runs it with ONNX Runtime on WebGPU or wasm.
3. **Tiles.** The image is cut into equal tiles, each read with an overlap of context mirrored past the
   borders as `realesrgan-ncnn-vulkan` does. Each row of tiles streams into a separable resampler, so memory
   follows the output size, not the 4× intermediate. Tiles paint live in the viewer as they finish.
4. **Output.** PNG goes through its own encoder (adaptive filters, `CompressionStream`), JPEG and WebP through
   `OffscreenCanvas`.

Checked against `realesrgan-ncnn-vulkan` 20220424: the same output to 45-55 dB PSNR (fp16 rounding and tile
seams), at about the same speed. A 1920 × 1080 anime cover to 3840 × 2160 takes ~27 s in Chrome on an Apple
M3, where ncnn-vulkan's 4× pass takes 22 s.

## Deployment

`Dockerfile` builds the static export (the checkpoints download in their own cached layer) and serves it
with nginx. `nginx.conf` sends the cross-origin isolation headers ONNX Runtime's wasm threads need
(`next.config.js` does the same for `next dev`). `deploy.sh` builds for arm64, ships the image to `midgar`,
and runs `docker compose up -d upscaler`.

## License

[MIT](LICENSE). Real-ESRGAN and its weights: BSD-3-Clause, © Xintao Wang. ONNX Runtime: MIT, © Microsoft, with
the notices of the third-party code in its wasm. The site serves these texts next to the weights
(`/models/LICENSE.txt`) and the runtime (`/ort/LICENSE.txt`, `/ort/ThirdPartyNotices.txt`), and shows them on
its Licenses page (`pages/licenses.tsx`).
