import Nav from '../sections/Nav';
import Hero from '../sections/Hero';
import StatsBar from '../sections/StatsBar';
import Features from '../sections/Features';
import Showcase from '../sections/Showcase';
import Performance from '../sections/Performance';
import Install from '../sections/Install';
import ModelMatrix from '../sections/ModelMatrix';
import Footer from '../sections/Footer';

export default function Home() {
  return (
    <div className="min-h-screen bg-[#0b0e13] text-[#e8ebf0] antialiased">
      <Nav />
      <main>
        <Hero />
        <StatsBar />
        <Features />
        <Showcase />
        <Performance />
        <Install />
        <ModelMatrix />
      </main>
      <Footer />
    </div>
  );
}
