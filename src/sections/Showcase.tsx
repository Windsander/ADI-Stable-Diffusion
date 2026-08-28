import { Section, SectionHead } from './shared';
import { showcaseItems } from '../content';

export default function Showcase() {
  return (
    <Section id="showcase">
      <SectionHead
        label="Showcase · 展厅"
        title="全部由 adi 本机生成"
        desc="以下每一帧都来自 adi CLI + ONNXRuntime 的本地 CPU 推理——无 Python、无 diffusers、无云端。静态图共用同一句 prompt：A cat in the water at sunset；SVD 将单张输入照片驱动为 14 帧影像。"
      />
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {showcaseItems.map((item) => (
          <figure
            key={item.model}
            className="group m-0 overflow-hidden rounded-lg border border-[rgba(255,255,255,0.08)] bg-[#10141b] transition-colors duration-200 hover:border-[rgba(62,198,221,0.35)]"
          >
            <div className="relative aspect-square overflow-hidden bg-[#0d1016]">
              {item.kind === 'image' ? (
                <img
                  src={item.media}
                  alt={item.model}
                  loading="lazy"
                  className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]"
                />
              ) : (
                <video
                  src={item.media}
                  autoPlay
                  muted
                  loop
                  playsInline
                  className="h-full w-full object-cover"
                />
              )}
              <span className="font-mono-ad absolute top-3 left-3 rounded border border-[rgba(255,255,255,0.14)] bg-[rgba(11,14,19,0.72)] px-2 py-1 text-[10px] tracking-[0.08em] text-[#c8d0dc] backdrop-blur-sm">
                {item.kind === 'video' ? 'VIDEO' : 'IMAGE'}
              </span>
            </div>
            <figcaption className="px-4 py-3.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono-ad text-[13px] font-semibold text-[#e8ebf0]">
                  {item.model}
                </span>
                <span className="font-mono-ad text-[11px] whitespace-nowrap text-[#5d6675]">
                  {item.resolution} · {item.steps}
                </span>
              </div>
              <div className="mt-1 text-[12px] text-[#9aa3b2]">{item.extra}</div>
            </figcaption>
          </figure>
        ))}
      </div>
    </Section>
  );
}
