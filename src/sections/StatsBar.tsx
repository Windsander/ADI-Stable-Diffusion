import { stats } from '../content';

export default function StatsBar() {
  return (
    <section className="border-t border-[rgba(255,255,255,0.08)] bg-[#0d1016]">
      <div className="mx-auto grid w-full max-w-[1120px] grid-cols-2 px-6 py-10 sm:grid-cols-3 md:grid-cols-5 md:px-8">
        {stats.map((s) => (
          <div key={s.unit} className="py-3 pr-4">
            <div className="font-mono-ad text-[26px] font-semibold text-[#e8ebf0] md:text-[30px]">
              {s.value}
            </div>
            <div className="mt-1 text-[12.5px] font-medium text-[#3ec6dd]">{s.unit}</div>
            <div className="mt-1.5 text-[11.5px] leading-[1.6] text-[#5d6675]">{s.note}</div>
          </div>
        ))}
      </div>
    </section>
  );
}
