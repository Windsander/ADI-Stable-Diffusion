import { Section, SectionHead } from './shared';
import { perfRows, perfNote, schedulerList } from '../content';

export default function Performance() {
  return (
    <Section id="performance">
      <SectionHead
        label="Performance · 性能"
        title="CPU 上的实测数字"
        desc="单次冷启动墙钟时间，未经任何预热与 GPU 加速——只有 adi CLI 和 ONNX 模型文件。"
      />

      <div className="overflow-hidden rounded-lg border border-[rgba(255,255,255,0.08)]">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-[rgba(255,255,255,0.08)] bg-[#0d1016]">
              {['模型', '分辨率', '步数', '墙钟时间'].map((h) => (
                <th
                  key={h}
                  className="font-mono-ad px-5 py-3.5 text-[11px] font-medium tracking-[0.14em] text-[#5d6675] uppercase"
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {perfRows.map((r) => (
              <tr
                key={r.model}
                className="border-b border-[rgba(255,255,255,0.06)] bg-[#10141b] transition-colors last:border-0 hover:bg-[#141a23]"
              >
                <td className="px-5 py-4 text-[13.5px] font-medium text-[#e8ebf0]">{r.model}</td>
                <td className="font-mono-ad px-5 py-4 text-[13px] text-[#9aa3b2]">{r.resolution}</td>
                <td className="font-mono-ad px-5 py-4 text-[13px] text-[#9aa3b2]">{r.steps}</td>
                <td className="font-mono-ad px-5 py-4 text-[14px] font-semibold text-[#3ec6dd]">
                  {r.time}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="font-mono-ad mt-3 text-[11.5px] text-[#5d6675]">{perfNote}</div>

      <div className="mt-14">
        <div className="font-mono-ad mb-5 text-[11px] font-medium tracking-[0.22em] text-[#5d6675] uppercase">
          Scheduler Arsenal · 采样器清单
        </div>
        <div className="flex flex-wrap gap-2">
          {schedulerList.map((s) => (
            <span
              key={s}
              className="font-mono-ad rounded border border-[rgba(255,255,255,0.1)] bg-[#10141b] px-3 py-1.5 text-[12px] text-[#9aa3b2] transition-colors hover:border-[rgba(62,198,221,0.4)] hover:text-[#3ec6dd]"
            >
              {s}
            </span>
          ))}
        </div>
        <p className="mt-4 text-[12.5px] leading-[1.8] text-[#5d6675]">
          全部与 HuggingFace diffusers 做轨迹级交叉验证；Karras σ（ρ=7）可与任意采样器组合。
        </p>
      </div>
    </Section>
  );
}
