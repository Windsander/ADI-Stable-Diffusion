import { useEffect, useState } from 'react';
import { Container } from './shared';
import { REPO_URL } from '../content';

const links = [
  { label: '能力', target: 'features' },
  { label: '展厅', target: 'showcase' },
  { label: '性能', target: 'performance' },
  { label: '安装', target: 'install' },
  { label: '矩阵', target: 'matrix' },
];

export default function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <header
      className={`fixed top-0 right-0 left-0 z-50 border-b transition-colors duration-300 ${
        scrolled
          ? 'border-[rgba(255,255,255,0.08)] bg-[rgba(11,14,19,0.82)] backdrop-blur-md'
          : 'border-transparent bg-transparent'
      }`}
    >
      <Container className="flex h-[60px] items-center justify-between">
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className="flex cursor-pointer items-center gap-2.5 border-none bg-transparent p-0"
        >
          <span className="flex h-7 w-7 items-center justify-center rounded-md border border-[rgba(62,198,221,0.4)] bg-[rgba(62,198,221,0.08)] font-serif text-[15px] text-[#3ec6dd]">
            σ
          </span>
          <span className="font-mono-ad text-[13px] font-semibold tracking-[0.06em] text-[#e8ebf0]">
            ADI
          </span>
          <span className="hidden text-[12px] text-[#5d6675] sm:inline">
            Agile Diffusers Inference
          </span>
        </button>

        <nav className="hidden items-center gap-7 md:flex">
          {links.map((l) => (
            <button
              key={l.target}
              onClick={() => go(l.target)}
              className="cursor-pointer border-none bg-transparent p-0 text-[13px] text-[#9aa3b2] transition-colors hover:text-[#e8ebf0]"
            >
              {l.label}
            </button>
          ))}
        </nav>

        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="font-mono-ad rounded-md border border-[rgba(255,255,255,0.14)] px-3.5 py-1.5 text-[12px] text-[#e8ebf0] no-underline transition-colors hover:border-[rgba(62,198,221,0.5)] hover:text-[#3ec6dd]"
        >
          GitHub ↗
        </a>
      </Container>
    </header>
  );
}
