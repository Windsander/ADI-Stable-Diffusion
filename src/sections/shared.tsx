import type { ReactNode } from 'react';

export function Container({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={`mx-auto w-full max-w-[1120px] px-6 md:px-8 ${className}`}>{children}</div>
  );
}

export function SectionHead({ label, title, desc }: { label: string; title: string; desc?: string }) {
  return (
    <div className="mb-10 md:mb-14">
      <div className="font-mono-ad mb-4 text-[11px] font-medium tracking-[0.22em] text-[#3ec6dd] uppercase">
        {label}
      </div>
      <h2 className="text-[28px] leading-[1.25] font-semibold tracking-[-0.01em] text-[#e8ebf0] md:text-[36px]">
        {title}
      </h2>
      {desc && (
        <p className="mt-4 max-w-[640px] text-[15px] leading-[1.9] text-[#9aa3b2]">{desc}</p>
      )}
    </div>
  );
}

export function Section({ id, children, className = '' }: { id?: string; children: ReactNode; className?: string }) {
  return (
    <section id={id} className={`border-t border-[rgba(255,255,255,0.08)] py-16 md:py-24 ${className}`}>
      <Container>{children}</Container>
    </section>
  );
}
