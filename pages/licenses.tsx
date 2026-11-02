import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import Head from 'next/head'

import type { GetStaticProps } from 'next'

type Props = { realesrgan: string; onnxruntime: string }

// The texts scripts/assets.ts wrote next to the weights and the ONNX Runtime files, read at build time
export const getStaticProps: GetStaticProps<Props> = () => {
  const read = (path: string) => readFileSync(join(process.cwd(), 'public', path), 'utf8')
  return { props: { realesrgan: read('models/LICENSE.txt'), onnxruntime: read('ort/LICENSE.txt') } }
}

const link = 'text-accent-strong underline decoration-accent/40 underline-offset-4 hover:decoration-accent'

const Licenses = ({ realesrgan, onnxruntime }: Props) => (
  <>
    <Head>
      <title>Licenses — Upscaler</title>
    </Head>
    <div className="h-full overflow-y-auto">
      <main className="text-fg-muted mx-auto max-w-3xl px-6 py-12 text-[13.5px] leading-relaxed">
        <a href="/" className="text-fg-faint hover:text-fg text-[12px]">
          ← Upscaler
        </a>
        <h1 className="text-fg mt-6 text-[22px] font-semibold tracking-tight">Licenses</h1>
        <p className="mt-3">
          Upscaler runs these works in your browser. It is not affiliated with or endorsed by their authors.
        </p>

        <section className="mt-10">
          <h2 className="text-fg text-[15px] font-semibold">Real-ESRGAN</h2>
          <p className="mt-2">
            The four models are converted from{' '}
            <a className={link} href="https://github.com/xinntao/Real-ESRGAN" target="_blank" rel="noreferrer">
              Real-ESRGAN
            </a>
            &apos;s released checkpoints and served from <code className="text-fg font-mono">/models/</code>, with this
            license as{' '}
            <a className={link} href="/models/LICENSE.txt">
              LICENSE.txt
            </a>
            .
          </p>
          <pre className="border-line bg-ink-1 text-fg-muted mt-4 overflow-x-auto rounded-lg border p-4 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap">
            {realesrgan}
          </pre>
        </section>

        <section className="mt-10">
          <h2 className="text-fg text-[15px] font-semibold">ONNX Runtime Web</h2>
          <p className="mt-2">
            The inference engine, served from <code className="text-fg font-mono">/ort/</code>. Its WebAssembly builds
            include third-party code under their own licenses, listed in its{' '}
            <a className={link} href="/ort/ThirdPartyNotices.txt">
              third-party notices
            </a>
            .
          </p>
          <pre className="border-line bg-ink-1 text-fg-muted mt-4 overflow-x-auto rounded-lg border p-4 font-mono text-[11.5px] leading-relaxed whitespace-pre-wrap">
            {onnxruntime}
          </pre>
        </section>

        <p className="border-line mt-10 border-t pt-6 text-[12.5px]">
          Upscaler&apos;s own code is MIT-licensed:{' '}
          <a className={link} href="https://github.com/verekia/upscaler" target="_blank" rel="noreferrer">
            source on GitHub
          </a>
          .
        </p>
      </main>
    </div>
  </>
)

export default Licenses
