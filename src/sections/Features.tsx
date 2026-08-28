import { Section, SectionHead } from './shared';
import { features } from '../content';

export default function Features() {
  return (
    <Section id="features">
      <SectionHead
        label="Capabilities · 能力"
        title="为工程部署而生"
        desc="设计目标只有三个：小包体积、格式兼容、数值忠实。以下每一项都在仓库里有对应的实现与验证记录。"
      />
      <div className="grid grid-cols-1 gap-px overflow-hidden rounded-lg border border-[rgba(255,255,255,0.08)] bg-[rgba(255,255,255,0.08)] sm:grid-cols-2 lg:grid-cols-3">
        {features.map((f) => (
          <div
            key={f.tag}
            className="bg-[#10141b] p-6 transition-colors duration-200 hover:bg-[#141a23] md:p-7"
          >
            <div className="font-mono-ad mb-4 text-[10.5px] font-medium tracking-[0.2em] text-[#3ec6dd]">
              {f.tag}
            </div>
            <h3 className="mb-2.5 text-[16.5px] font-semibold text-[#e8ebf0]">{f.title}</h3>
            <p className="m-0 text-[13.5px] leading-[1.85] text-[#9aa3b2]">{f.body}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
