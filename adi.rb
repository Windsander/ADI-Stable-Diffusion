class adi < Formula
  desc "ADI Stable Diffusion"
  homepage "https://github.com/Windsander/ADI-Stable-Diffusion"
  version "v1.0.1"
  license "GPL-3.0 license"

  if Hardware::CPU.intel?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v1.0.1/release-v1.0.1-macos-x86_64.tar.gz"
    sha256 "894be3a3230ef75326b5c95315b770d0cc33a904401aa94957af1b7f0484021d"
  elsif Hardware::CPU.arm?
    url "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-v1.0.1/release-v1.0.1-macos-arm64.tar.gz"
    sha256 "bfc0dfc7ace4b6ded471a385fe69465fb9928d578db8a8c8745c8f1b0559028d"
  else
    odie "Unsupported architecture"
  end


  def install
    # 安装可执行文件和动态库到bin目录
    bin.install Dir["bin/*"]

    # 安装头文件到include目录
    include.install Dir["include/*"]

    # 安装静态库和动态库到lib目录
    lib.install Dir["lib/*"]

    # 安装其他文件
    prefix.install "CHANGELOG.md"
    prefix.install "README.md"
    prefix.install "LICENSE"
  end

  test do
    # 运行测试来验证安装是否成功
    system "#{bin}/adi", "--version"
  end
end
