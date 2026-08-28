import { Section, SectionHead } from './shared';
import { matrixRows } from '../content';

export default function ModelMatrix() {
  return (
    <Section id="matrix">
      <SectionHead
        label="Model Matrix · 参数矩阵"
        title="每个家族的确切参数面"
        desc="从 SD1.x 到 FLUX / SVD，六条家族线的关键参数一行到位——复制到 CLI 即可直接跑。"
      />

      <div className="overflow-x-auto rounded-lg border border-[rgba(255,255,255,0.08)]">
        <table className="w-full min-w-[760px] border-collapse text-left">
          <thead>
            <tr className="border-b border-[rgba(255,255,255,0.08)] bg-[#0d1016]">
              {['模型', '--dims', '--predictor', '--decoding', '备注'].map((h) => (
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
            {matrixRows.map((r) => (
              <tr
                key={r.model}
                className="border-b border-[rgba(255,255,255,0.06)] bg-[#10141b] transition-colors last:border-0 hover:bg-[#141a23]"
              >
                <td className="px-5 py-4 text-[13.5px] font-medium whitespace-nowrap text-[#e8ebf0]">
                  {r.model}
                </td>
                <td className="font-mono-ad px-5 py-4 text-[13px] whitespace-nowrap text-[#3ec6dd]">
                  {r.dims}
                </td>
                <td className="font-mono-ad px-5 py-4 text-[13px] whitespace-nowrap text-[#9aa3b2]">
                  {r.predictor}
                </td>
                <td className="font-mono-ad px-5 py-4 text-[13px] whitespace-nowrap text-[#9aa3b2]">
                  {r.decoding}
                </td>
                <td className="px-5 py-4 text-[12.5px] leading-[1.7] text-[#5d6675]">{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-[12.5px] leading-[1.8] text-[#5d6675]">
        完整参数说明与更多示例，见仓库 README 的
        <span className="font-mono-ad"> usage </span>
        章节与 <span className="font-mono-ad">docs/</span> 目录。
      </p>
    </Section>
  );
}
