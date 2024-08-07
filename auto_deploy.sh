#!/bin/bash

set -e

# Function to display usage
usage() {
    echo "Usage: $0 --version [version]"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --version)
        VERSION="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        usage # unknown option
        ;;
    esac
done

# Check if VERSION is set
if [[ -z "$VERSION" ]]; then
    echo "Error: --version [version] is required"
    usage
fi

echo "Deploying version: $VERSION"

# Static variables
REPO_URL="https://github.com/Windsander/ADI-Stable-Diffusion"
DESCRIPTION="Agile Diffusers Inference (ADI) Command Line Tool"
LONG_DESCRIPTION="Agile Diffusers Inference (ADI) is a C++ project. Its purpose is to leverage the acceleration capabilities of ONNXRuntime and the high compatibility of the .onnx model format to provide a convenient solution for the engineering deployment of Stable Diffusion."
MAINTAINER="Arikan.Li<https://github.com/Windsander/ADI-Stable-Diffusion/issues>"
LICENSE="GPL-3.0 license"

# Ensure necessary tools are installed
ensure_tools() {
    echo "Ensuring necessary tools are installed..."

    OS=$(uname -s)

    case "$OS" in
        Linux*)
            # Ensure tools for Linux (Ubuntu)
            if ! command -v cmake &> /dev/null; then
                echo "CMake not found, installing..."
                sudo apt-get update
                sudo apt-get install -y cmake
            fi
            if ! command -v ninja &> /dev/null; then
                echo "Ninja not found, installing..."
                sudo apt-get install -y ninja-build
            fi
            if ! command -v dput &> /dev/null; then
                echo "dput not found, installing..."
                sudo apt-get install -y dput
            fi
            if ! command -v copr-cli &> /dev/null; then
                echo "copr-cli not found, installing..."
                sudo pip3 install copr-cli
            fi
            if ! command -v curl &> /dev/null; then
                echo "curl not found, installing..."
                sudo apt-get install -y curl
            fi
            if ! command -v sha256sum &> /dev/null; then
                echo "sha256sum not found, installing..."
                sudo apt-get install -y coreutils
            fi
            if ! command -v dpkg-deb &> /dev/null; then
                echo "dpkg-deb not found, installing..."
                sudo apt-get install -y dpkg
            fi
            if ! command -v debuild &> /dev/null; then
                echo "debuild not found, installing..."
                sudo apt-get install -y devscripts
            fi
            if ! command -v rpmbuild &> /dev/null; then
                echo "rpmbuild not found, installing..."
                sudo apt-get install -y rpm
            fi
            ;;

        Darwin*)
            # Ensure tools for macOS
            if ! command -v cmake &> /dev/null; then
                echo "CMake not found, installing..."
                brew install cmake
            fi
            if ! command -v ninja &> /dev/null; then
                echo "Ninja not found, installing..."
                brew install ninja
            fi
            if ! command -v curl &> /dev/null; then
                echo "curl not found, installing..."
                brew install curl
            fi
            if ! command -v sha256sum &> /dev/null; then
                echo "sha256sum not found, installing..."
                brew install coreutils
            fi
            # Note: dput, copr-cli, dpkg-deb, debuild, and rpmbuild are not typical on macOS
            ;;

        CYGWIN*|MINGW*|MSYS*)
            # Ensure tools for Windows (via Chocolatey)
            if ! command -v choco &> /dev/null; then
                echo "Chocolatey not found, installing..."
                powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"
            fi
            if ! command -v cmake &> /dev/null; then
                echo "CMake not found, installing..."
                choco install cmake -y
            fi
            if ! command -v ninja &> /dev/null; then
                echo "Ninja not found, installing..."
                choco install ninja -y
            fi
            if ! command -v curl &> /dev/null; then
                echo "curl not found, installing..."
                choco install curl -y
            fi
            if ! command -v sha256sum &> /dev/null; then
                echo "sha256sum not found, installing..."
                choco install coreutils -y
            fi
            # Note: dput, copr-cli, dpkg-deb, debuild, and rpmbuild are not typical on Windows
            ;;
    esac
}

# Create Homebrew Formula
create_homebrew_formula() {
  echo "Creating Homebrew Formula..."

  local formula_name=$1
  local version=$2
  local url_x86_64=$3
  local url_arm64=$4

  # 计算 SHA-256 校验和
  local sha256_x86_64
  sha256_x86_64=$(curl -L ${url_x86_64} | shasum -a 256 | awk '{ print $1 }')

  local sha256_arm64
  sha256_arm64=$(curl -L ${url_arm64} | shasum -a 256 | awk '{ print $1 }')

  cat <<EOF > ${formula_name}-${version}.rb
class ${formula_name^} < Formula
  desc "ADI Stable Diffusion"
  homepage "https://github.com/Windsander/ADI-Stable-Diffusion"
  version "${version}"
  license "${LICENSE}"

  if Hardware::CPU.intel?
    url "${url_x86_64}"
    sha256 "${sha256_x86_64}"
  elsif Hardware::CPU.arm?
    url "${url_arm64}"
    sha256 "${sha256_arm64}"
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
    system "#{bin}/ort-sd-clitools", "--version"
  end
end
EOF

  echo "Formula made: ${formula_name}-${version}.rb"

  # 创建指向最新版本公式文件的符号链接，并覆盖上一个版本的链接 (以求方便用户直接安装最新版)
  ln -sf ${formula_name}-${version}.rb ${formula_name}.rb
  echo "Linking last: ${formula_name}.rb -> ${formula_name}-${version}.rb"

  echo "Formula created successfully"
}

# Create Debian Package
create_debian_package() {
  echo "Creating Debian Package..."

  local package_name=$1
  local version=$2
  local url_x86_64=$3
  local url_aarch64=$4

  # 计算 SHA-256 校验和
  local sha256_x86_64
  sha256_x86_64=$(curl -L ${url_x86_64} | sha256sum | awk '{ print $1 }')

  local sha256_aarch64
  sha256_aarch64=$(curl -L ${url_aarch64} | sha256sum | awk '{ print $1 }')

  # 创建临时目录结构
  mkdir -p ${package_name}-${version}/debian

  # 创建 debian/changelog 文件
  cat <<EOF > ${package_name}-${version}/debian/changelog
${package_name} (${version}-1) stable; urgency=low

  * See CHANGELOG.md in package

 -- $MAINTAINER  $(date +"%a, %d %b %Y %H:%M:%S %z")
EOF

  # 创建 debian/rules 文件
  cat <<EOF > ${package_name}-${version}/debian/rules
#!/usr/bin/make -f

# 定义变量
PACKAGE_NAME := ${package_name}
VERSION := ${version}
ARCH := \$(shell dpkg-architecture -qDEB_HOST_ARCH)
URL_X86_64 := ${url_x86_64}
URL_AARCH64 := ${url_aarch64}
SHA256_X86_64 := ${sha256_x86_64}
SHA256_AARCH64 := ${sha256_aarch64}

%:
	dh \$@

override_dh_auto_build:
	if [ "\$(ARCH)" = "amd64" ]; then \
		curl -L -o release-\$(VERSION)-linux-x86_64.tar.gz \$(URL_X86_64); \
		echo "\$(SHA256_X86_64)  release-\$(VERSION)-linux-x86_64.tar.gz" | sha256sum -c -; \
		tar -xzvf release-\$(VERSION)-linux-x86_64.tar.gz; \
	elif [ "\$(ARCH)" = "arm64" ]; then \
		curl -L -o release-\$(VERSION)-linux-aarch64.tar.gz \$(URL_AARCH64); \
		echo "\$(SHA256_AARCH64)  release-\$(VERSION)-linux-aarch64.tar.gz" | sha256sum -c -; \
		tar -xzvf release-\$(VERSION)-linux-aarch64.tar.gz; \
	fi

override_dh_auto_install:
	mkdir -p \$(DESTDIR)/usr/local/bin
  mkdir -p \$(DESTDIR)/usr/local/include
  mkdir -p \$(DESTDIR)/usr/local/lib
  mkdir -p \$(DESTDIR)/usr/share/doc/\$(PACKAGE_NAME)
  cp -r bin/* \$(DESTDIR)/usr/local/bin/
  cp -r include/* \$(DESTDIR)/usr/local/include/
  cp -r lib/* \$(DESTDIR)/usr/local/lib/
  cp CHANGELOG.md \$(DESTDIR)/usr/share/doc/\$(PACKAGE_NAME)/
  cp README.md \$(DESTDIR)/usr/share/doc/\$(PACKAGE_NAME)/
  cp LICENSE \$(DESTDIR)/usr/share/doc/\$(PACKAGE_NAME)/
EOF

  chmod +x ${package_name}-${version}/debian/rules

  # 创建 debian/control 文件
  cat <<EOF > ${package_name}-${version}/debian/control
Source: ${package_name}
Section: base
Priority: optional
Maintainer: ${MAINTAINER}
Build-Depends: debhelper-compat (= 13), curl, sha256sum
Standards-Version: 4.5.0
Homepage: ${REPO_URL}
Rules-Requires-Root: no

Package: ${package_name}
Architecture: any
Depends: \${misc:Depends}
Description: ${DESCRIPTION}
 ${LONG_DESCRIPTION}
EOF

  # 创建空的 bin, include 和 lib 目录
  mkdir -p ${package_name}-${version}/bin
  mkdir -p ${package_name}-${version}/include
  mkdir -p ${package_name}-${version}/lib

  # 打包 debian 包
  cd ${package_name}-${version}
  debuild -us -uc
  cd ..

  # 重命名生成的 deb 文件
  mv ../${package_name}_${version}_*.deb ${package_name}-${version}.deb

  # 清理临时目录
  rm -rf ${package_name}-${version}

  echo "Debian packages made: ${package_name}-${version}.deb"
  echo "Debian packages created successfully"
}

# Create RPM Package
create_rpm_package() {
  echo "Creating RPM Package..."

  local package_name=$1
  local version=$2
  local url_x86_64=$3
  local url_aarch64=$4

  # 计算 SHA-256 校验和
  local sha256_x86_64
  sha256_x86_64=$(curl -L ${url_x86_64} | sha256sum | awk '{ print $1 }')

  local sha256_aarch64
  sha256_aarch64=$(curl -L ${url_aarch64} | sha256sum | awk '{ print $1 }')

  # 创建临时目录
  mkdir -p ${package_name}-${version}/BUILD
  mkdir -p ${package_name}-${version}/RPMS
  mkdir -p ${package_name}-${version}/SOURCES
  mkdir -p ${package_name}-${version}/SPECS
  mkdir -p ${package_name}-${version}/SRPMS

  # 创建通用 SPEC 文件
  cat <<EOF > ${package_name}-${version}/SPECS/${package_name}.spec
%define name ${package_name}
%define version ${version}
%define sha256_x86_64 ${sha256_x86_64}
%define sha256_aarch64 ${sha256_aarch64}

Name: %{name}
Version: %{version}
Release: 1%{?dist}
Summary: ${DESCRIPTION}

License: ${LICENSE}
URL: ${REPO_URL}

%description
${LONG_DESCRIPTION}

%ifarch x86_64
Source0: ${url_x86_64}
%endif
%ifarch aarch64
Source0: ${url_aarch64}
%endif

%prep
%ifarch x86_64
%define expected_sha256sum %{sha256_x86_64}
%endif
%ifarch aarch64
%define expected_sha256sum %{sha256_aarch64}
%endif

curl -L -o %{_sourcedir}/%{name}-%{version}-%{arch}.tar.gz %{SOURCE0}
echo "%{expected_sha256sum}  %{_sourcedir}/%{name}-%{version}-%{arch}.tar.gz" | sha256sum -c -
%setup -q -n %{name}-%{version}-%{arch}

%build

%install
mkdir -p %{buildroot}/usr/local/bin
mkdir -p %{buildroot}/usr/local/include
mkdir -p %{buildroot}/usr/local/lib
cp -r * %{buildroot}/usr/local/

%files
/usr/local/bin/*
/usr/local/include/*
/usr/local/lib/*
%doc CHANGELOG.md README.md LICENSE

%changelog
* $(date +"%a %b %d %Y") ${MAINTAINER} - ${version}-1
- See CHANGELOG.md in package
EOF

  # 打包 x86_64 rpm
  rpmbuild --define "_topdir $(pwd)/${package_name}-${version}" --target x86_64 -ba ${package_name}-${version}/SPECS/${package_name}.spec
  mv ${package_name}-${version}/RPMS/x86_64/${package_name}-${version}-1.x86_64.rpm ${package_name}-${version}-x86_64.rpm

  # 打包 aarch64 rpm
  rpmbuild --define "_topdir $(pwd)/${package_name}-${version}" --target aarch64 -ba ${package_name}-${version}/SPECS/${package_name}.spec
  mv ${package_name}-${version}/RPMS/aarch64/${package_name}-${version}-1.aarch64.rpm ${package_name}-${version}-aarch64.rpm

  # 清理临时目录
  rm -rf ${package_name}-${version}

  echo "RPM packages made: ${package_name}-${version}-x86_64[aarch64].rpm"
  echo "RPM packages created successfully"
}

# Main function
main() {
    ensure_tools

    echo "==========================================================="

    create_homebrew_formula "adi" "${VERSION}" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-macos-x86_64.tar.gz" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-macos-arm64.tar.gz"

    echo "==========================================================="

    create_debian_package "adi" "${VERSION}" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-x86_64.tar.gz" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-aarch64.tar.gz"

    echo "==========================================================="

    create_rpm_package "adi" "${VERSION}" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-x86_64.tar.gz" \
      "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-aarch64.tar.gz"

}

main

echo "Deployment completed successfully."