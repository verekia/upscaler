/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  reactCompiler: true,
  output: 'export',
  allowedDevOrigins: ['upscaler.dev'],
  // Cross-origin isolation gives ONNX Runtime's wasm its threads (SharedArrayBuffer). Only for `next dev`:
  // the static export gets the same headers from nginx.conf (and `next build` warns about any headers).
  ...(process.env.NODE_ENV === 'development' && {
    headers: async () => [
      {
        source: '/:path*',
        headers: [
          { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ],
  }),
}

export default nextConfig
