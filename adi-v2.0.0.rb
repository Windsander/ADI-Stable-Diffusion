class Adi < Formula
  desc "ADI Stable Diffusion"
  homepage "https://github.com/Windsander/ADI-Stable-Diffusion"
  version "v2.0.0"
  license "GPL-3.0 license"

  if Hardware::CPU.intel?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v2.0.0/release-v2.0.0-macos-x86_64.tar.gz"
    sha256 "0019dfc4b32d63c1392aa264aed2253c1e0c2fb09216f8e2cc269bbfb8bb49b5"
  elsif Hardware::CPU.arm?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v2.0.0/release-v2.0.0-macos-arm64.tar.gz"
    sha256 "5fc986cd7476558a1b9cbe16490a6c50810d9871c2507d301ab90711c04319d5"
  else
    odie "Unsupported architecture"
  end


  def install
    bin.install Dir["bin/*"]
    lib.install Dir["lib/*"]
    include.install Dir["include/*"]

    prefix.install "CHANGELOG.md"
    prefix.install "README.md"
    prefix.install "LICENSE"
  end

  test do
    system "#{bin}/adi", "--version"
  end
end
