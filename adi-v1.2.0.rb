class Adi < Formula
  desc "ADI Stable Diffusion"
  homepage "https://github.com/Windsander/ADI-Stable-Diffusion"
  version "v1.2.0"
  license "GPL-3.0 license"

  if Hardware::CPU.intel?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v1.2.0/release-v1.2.0-macos-x86_64.tar.gz"
    sha256 "448ec19467c07f63918f02b5e646c700d2b8b52abf7429ea2375088b664b6e85"
  elsif Hardware::CPU.arm?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v1.2.0/release-v1.2.0-macos-arm64.tar.gz"
    sha256 "f67bb433110518cd49b831516f26b02ab7f02ed3c457436e8c1d3713d74d5f1f"
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
