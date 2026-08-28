import { Container } from './shared';
import { REPO_URL } from '../content';

const termLines = [
  { text: 'adi -p "A cat in the water at sunset" -m txt2img \\', prompt: true },
  { text: '    --unet onnx-flux-schnell/transformer/model.onnx \\', prompt: false },
  { text: '    --scheduler flow_euler --steps 4 --precision auto', prompt: false },
];

export default function Hero() {
  return (
    <section className="relative overflow-hidden pt-[132px] pb-16 md:pt-[168px] md:pb-24">
      {/* dot grid + top glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.055) 1px, transparent 1px)',
          backgroundSize: '26px 26px',
          maskImage: 'radial-gradient(ellipse 90% 70% at 50% 0%, black 30%, transparent 75%)',
          WebkitMaskImage: 'radial-gradient(ellipse 90% 70% at 50% 0%, black 30%, transparent 75%)',
        }}
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse 55% 40% at 50% -6%, rgba(62,198,221,0.13), transparent 70%)',
        }}
      />

      <Container className="relative">
        <div className="font-mono-ad mb-6 flex flex-wrap items-center gap-x-3 gap-y-2 text-[11px] tracking-[0.18em] text-[#5d6675] uppercase">
          <span className="text-[#3ec6dd]">Agile Diffusers Inference</span>
          <span>·</span>
          <span>v2.0.0</span>
          <span>·</span>
          <span>ONNXRuntime 1.28</span>
          <span>·</span>
          <span>GPL-3.0</span>
        </div>

        <h1 className="max-w-[760px] text-[38px] leading-[1.18] font-bold tracking-[-0.02em] text-[#e8ebf0] md:text-[56px]">
          纯 C++ 的 Stable Diffusion
          <span className="text-[#3ec6dd]"> 推理引擎</span>
        </h1>

        <p className="mt-6 max-w-[620px] text-[15px] leading-[1.95] text-[#9aa3b2] md:text-[16px]">
          零 Python 推理时，一条 CLI，九个平台目标。从 SD v1.5 到 SD3.5 / FLUX / SVD——
          选一个采样器，指向 ONNX 模型，开始生成。
        </p>

        <div className="mt-9 flex flex-wrap items-center gap-3">
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-md bg-[#3ec6dd] px-5 py-2.5 text-[13.5px] font-semibold text-[#0b0e13] no-underline transition-colors hover:bg-[#5cd4e6]"
          >
            GitHub 仓库
          </a>
          <button
            onClick={() => document.getElementById('install')?.scrollIntoView({ behavior: 'smooth' })}
            className="cursor-pointer rounded-md border border-[rgba(255,255,255,0.14)] bg-transparent px-5 py-2.5 text-[13.5px] font-medium text-[#e8ebf0] transition-colors hover:border-[rgba(62,198,221,0.5)] hover:text-[#3ec6dd]"
          >
            快速开始
          </button>
        </div>

        {/* terminal */}
        <div className="mt-14 overflow-hidden rounded-lg border border-[rgba(255,255,255,0.1)] bg-[#0d1117] shadow-[0_24px_80px_rgba(0,0,0,0.45)]">
          <div className="flex items-center gap-1.5 border-b border-[rgba(255,255,255,0.07)] px-4 py-3">
            <span className="h-2.5 w-2.5 rounded-full bg-[#2b333f]" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#2b333f]" />
            <span className="h-2.5 w-2.5 rounded-full bg-[#2b333f]" />
            <span className="font-mono-ad ml-3 text-[11px] text-[#5d6675]">zsh — adi</span>
          </div>
          <div className="font-mono-ad overflow-x-auto px-5 py-5 text-[12.5px] leading-[2] whitespace-pre">
            {termLines.map((l, i) => (
              <div key={i}>
                {l.prompt && <span className="mr-2 text-[#3ec6dd]">$</span>}
                <span className="text-[#c8d0dc]">{l.text}</span>
              </div>
            ))}
            <div className="mt-2 text-[#5d6675]">
              <span className="text-[#3ec6dd]">✓</span> 1024×1024 · 4 steps · flow_euler · CPU only — done.
            </div>
          </div>
        </div>
      </Container>
    </section>
  );
}
