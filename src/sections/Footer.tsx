import { Container } from './shared';
import { footerColumns, REPO_URL } from '../content';

export default function Footer() {
  return (
    <footer className="border-t border-[rgba(255,255,255,0.08)]">
      {/* 愿景区 */}
      <Container className="py-16 md:py-24">
        <div className="font-mono-ad mb-5 text-[11px] font-medium tracking-[0.22em] text-[#3ec6dd] uppercase">
          Vision · 愿景
        </div>
        <p className="max-w-[720px] text-[26px] leading-[1.4] font-semibold tracking-[-0.01em] text-[#e8ebf0] md:text-[34px]">
          把扩散模型，
          <span className="text-[#3ec6dd]">装进任何设备</span>。
        </p>
        <p className="mt-5 max-w-[560px] text-[14px] leading-[1.9] text-[#9aa3b2]">
          没有 Python 运行时，没有云端依赖——一个头文件、六个 C ABI 入口，从服务器到嵌入式，同一条推理路径。
        </p>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="mt-8 inline-flex items-center gap-2 rounded-md border border-[rgba(62,198,221,0.4)] bg-[rgba(62,198,221,0.08)] px-5 py-2.5 text-[13.5px] font-medium text-[#3ec6dd] transition-colors hover:bg-[rgba(62,198,221,0.16)]"
        >
          <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden>
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
          </svg>
          在 GitHub 上参与开发
        </a>
      </Container>

      {/* 链接区 */}
      <div className="border-t border-[rgba(255,255,255,0.08)] bg-[#0d1016]">
        <Container className="py-12">
          <div className="grid grid-cols-2 gap-10 md:grid-cols-4">
            <div>
              <div className="flex items-center gap-2.5">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <rect x="1.5" y="1.5" width="21" height="21" rx="5" stroke="#3ec6dd" strokeWidth="1.4" />
                  <text x="12" y="16.5" textAnchor="middle" fontSize="11" fontFamily="JetBrains Mono, monospace" fill="#3ec6dd" fontWeight="600">σ</text>
                </svg>
                <span className="text-[14px] font-semibold text-[#e8ebf0]">ADI</span>
              </div>
              <p className="mt-3 text-[12px] leading-[1.8] text-[#5d6675]">
                Agile Diffusers Inference
                <br />
                纯 C++ 的 Stable Diffusion 推理引擎
              </p>
            </div>
            {footerColumns.map((col) => (
              <div key={col.heading}>
                <div className="font-mono-ad mb-4 text-[10.5px] font-medium tracking-[0.2em] text-[#5d6675] uppercase">
                  {col.heading}
                </div>
                <ul className="space-y-2.5">
                  {col.links.map((l) => (
                    <li key={l.text}>
                      <a
                        href={l.href}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[13px] text-[#9aa3b2] transition-colors hover:text-[#3ec6dd]"
                      >
                        {l.text}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div className="font-mono-ad mt-12 flex flex-col gap-2 border-t border-[rgba(255,255,255,0.06)] pt-6 text-[11.5px] text-[#5d6675] md:flex-row md:items-center md:justify-between">
            <span>© 2026 Windsander · ADI is open source under GPL-3.0.</span>
            <span>adi.cyberfederal.io · Built with React + Vite</span>
          </div>
        </Container>
      </div>
    </footer>
  );
}
