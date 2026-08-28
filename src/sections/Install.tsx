import { Section, SectionHead } from './shared';
import { installMethods } from '../content';

export default function Install() {
  return (
    <Section id="install">
      <SectionHead
        label="Quick Start · 安装"
        title="三分钟跑起来"
        desc="包管理器一键安装，或者从 Releases 直接下载 bin / lib / include 三件套，也可以源码交叉编译到任何目标平台。"
      />
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {installMethods.map((m) => (
          <div
            key={m.id}
            className="overflow-hidden rounded-lg border border-[rgba(255,255,255,0.08)] bg-[#0d1117]"
          >
            <div className="flex items-center justify-between border-b border-[rgba(255,255,255,0.07)] px-4 py-3">
              <span className="font-mono-ad text-[11px] tracking-[0.1em] text-[#9aa3b2]">
                {m.label}
              </span>
              <span className="h-1.5 w-1.5 rounded-full bg-[#3ec6dd]" />
            </div>
            <div
              className="font-mono-ad overflow-x-auto px-4 py-4 text-[12px] leading-[2.1] whitespace-pre"
              style={{
                maskImage: 'linear-gradient(to right, black calc(100% - 32px), transparent)',
                WebkitMaskImage: 'linear-gradient(to right, black calc(100% - 32px), transparent)',
              }}
            >
              {m.lines.map((line, i) => (
                <div key={i}>
                  <span className="mr-2 text-[#3ec6dd]">$</span>
                  <span className="text-[#c8d0dc]">{line}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-5 text-[12.5px] leading-[1.8] text-[#5d6675]">
        注：包管理器渠道当前提供 v1.0.1；v2.0.0 各平台产物请从 Releases 页面获取（由 release/release-v* 分支的自动化链产出）。
      </p>
    </Section>
  );
}
