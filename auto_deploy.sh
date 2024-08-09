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
LONG_DESCRIPTION="Agile Diffusers Inference (ADI) is a C++ library and CLI tool that leverages ONNXRuntime for efficient deployment of Stable Diffusion models, offering high performance and compact size."
MAINTAINER="Arikan.Li <arikanli@cyberfederal.io>"
LICENSE="GPL-3.0 license"
LICENSE_URL="https://www.gnu.org/licenses/gpl-3.0.en.html"

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
                sudo apt-get install -y python3-pip
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
            if ! command -v brew &> /dev/null; then
                echo "Homebrew not found, please install Homebrew first."
                exit 1
            fi
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
                choco install git -y
            fi
            ;;
    esac

    echo "Ensuring necessary tools are installed [Finish]"
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
class ${formula_name} < Formula
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
    system "#{bin}/adi", "--version"
  end
end
EOF

  echo "Formula made: ${formula_name}-${version}.rb"

  # 创建指向最新版本公式文件的符号链接，并覆盖上一个版本的链接 (以求方便用户直接安装最新版)
  ln -sf ${formula_name}-${version}.rb ${formula_name}.rb
  echo "Formula link: ${formula_name}.rb -> ${formula_name}-${version}.rb"

  echo "Formula created successfully"
}

# Create Debian Package
create_debian_package() {
  echo "Creating Debian Package..."

  local package_name=$1
  local version=${2#v}
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
${package_name} (${version}-1) UNRELEASED; urgency=low

  * See CHANGELOG.md in package

 -- $MAINTAINER  $(date -R)
EOF

  # 创建 debian/rules 文件
  cat <<EOF > ${package_name}-${version}/debian/rules
#!/usr/bin/make -f

%:
	dh \$@

override_dh_auto_install:
	# 创建必要的目录
	mkdir -p \$(CURDIR)/debian/${package_name}/usr/bin
	mkdir -p \$(CURDIR)/debian/${package_name}/usr/include
	mkdir -p \$(CURDIR)/debian/${package_name}/usr/lib
	mkdir -p \$(CURDIR)/debian/${package_name}/usr/share/doc/${package_name}

	# 复制二进制文件
	cp -r \$(CURDIR)/bin/* \$(CURDIR)/debian/${package_name}/usr/bin/ || true

	# 复制头文件
	cp -r \$(CURDIR)/include/* \$(CURDIR)/debian/${package_name}/usr/include/ || true

	# 复制库文件
	cp -r \$(CURDIR)/lib/* \$(CURDIR)/debian/${package_name}/usr/lib/ || true

	# 复制文档
	cp \$(CURDIR)/CHANGELOG.md \$(CURDIR)/debian/${package_name}/usr/share/doc/${package_name}/ || true
	cp \$(CURDIR)/README.md \$(CURDIR)/debian/${package_name}/usr/share/doc/${package_name}/ || true
	cp \$(CURDIR)/LICENSE \$(CURDIR)/debian/${package_name}/usr/share/doc/${package_name}/ || true

	# 安装库文件并创建符号链接
	install -D -m 644 \$(CURDIR)/lib/libonnxruntime.so \$(CURDIR)/debian/${package_name}/usr/lib/libonnxruntime.so
	ln -sf libonnxruntime.so \$(CURDIR)/debian/${package_name}/usr/lib/libonnxruntime.so.1
	ln -sf libonnxruntime.so \$(CURDIR)/debian/${package_name}/usr/lib/libonnxruntime.so.1.18.0
EOF

  chmod +x ${package_name}-${version}/debian/rules

  # 创建 debian/control 文件
  cat <<EOF > ${package_name}-${version}/debian/control
Source: ${package_name}
Section: utils
Priority: optional
Maintainer: ${MAINTAINER}
Build-Depends: debhelper-compat (= 13), curl
Standards-Version: 4.5.0
Rules-Requires-Root: no
Homepage: ${REPO_URL}

Package: ${package_name}
Architecture: any
Depends: \${misc:Depends}
Description: ${DESCRIPTION}
 ${LONG_DESCRIPTION}
EOF

  # 创建 debian/source/format 文件
  mkdir -p ${package_name}-${version}/debian/source
  cat <<EOF > ${package_name}-${version}/debian/source/format
3.0 (quilt)
EOF

  # 创建 debian/source/include-binaries 文件
  cat <<EOF > ${package_name}-${version}/debian/source/include-binaries
bin/${package_name}
lib/lib${package_name}.a
lib/libonnxruntime.so
EOF

  # 创建 debian/copyright 文件
  cat <<EOF > ${package_name}-${version}/debian/copyright
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: ${package_name}
Source: ${REPO_URL}

Files: *
Copyright: 2023 Arikan.Li
License: GPL-3.0+
 This package is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.
 .
 This package is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 .
 On Debian systems, the complete text of the GNU General Public License
 version 3 can be found in '/usr/share/common-licenses/GPL-3'.
EOF

  # 数据准备
  ls -la

  curl -L -o release-${version}-linux-x86_64.tar.gz ${url_x86_64}
  echo "${sha256_x86_64}  release-${version}-linux-x86_64.tar.gz" | sha256sum -c -
  mkdir -p ${package_name}-${version}-x86_64.orig
  tar -xzvf release-${version}-linux-x86_64.tar.gz -C ${package_name}-${version}-x86_64.orig --strip-components=1

  curl -L -o release-${version}-linux-aarch64.tar.gz ${url_aarch64}
  echo "${sha256_aarch64}  release-${version}-linux-aarch64.tar.gz" | sha256sum -c -
  mkdir -p ${package_name}-${version}-aarch64.orig
  tar -xzvf release-${version}-linux-aarch64.tar.gz -C ${package_name}-${version}-aarch64.orig --strip-components=1

  # 环境准备
  ls -la

  mkdir -p ${package_name}-${version}-x86_64
  cp -r ${package_name}-${version}/debian ${package_name}-${version}-x86_64/
  cp -r ${package_name}-${version}-x86_64.orig/* ${package_name}-${version}-x86_64/

  mkdir -p ${package_name}-${version}-aarch64
  cp -r ${package_name}-${version}/debian ${package_name}-${version}-aarch64/
  cp -r ${package_name}-${version}-aarch64.orig/* ${package_name}-${version}-aarch64/

  # 打包 x86_64 架构
  echo "enter: ${package_name}-${version}-x86_64"
  tar -czf ${package_name}_${version}.orig.tar.gz -C ${package_name}-${version}-x86_64.orig .
  cd ${package_name}-${version}-x86_64
  fakeroot debuild -us -uc && ls -la
  cd ..
  rm -f ${package_name}_${version}.orig.tar.gz
  echo "back to buildroot"

  # 打包 aarch64 架构
  echo "enter: ${package_name}-${version}-aarch64"
  tar -czf ${package_name}_${version}.orig.tar.gz -C ${package_name}-${version}-aarch64.orig .
  cd ${package_name}-${version}-aarch64
  fakeroot debuild -us -uc && ls -la
  cd ..
  rm -f ${package_name}_${version}.orig.tar.gz
  echo "back to buildroot"

  # 重命名 & 清理过程资源
  mv ${package_name}_${version}-1_amd64.deb ${package_name}-${version}-x86_64.deb
  rm -rf ${package_name}-${version}-x86_64.orig
  rm -rf ${package_name}-${version}-x86_64

  mv ${package_name}_${version}-1_arm64.deb ${package_name}-${version}-aarch64.deb
  rm -rf ${package_name}-${version}-aarch64.orig
  rm -rf ${package_name}-${version}-aarch64

  ls -la

  echo "Debian packages created successfully"
}

# Create RPM Package
create_rpm_package() {
  echo "Creating RPM Package..."

  local package_name=$1
  local version=$2
  local url_x86_64=$3
  local url_aarch64=$4

  mkdir -p ${package_name}-${version}/BUILD
  mkdir -p ${package_name}-${version}/RPMS
  mkdir -p ${package_name}-${version}/SPECS
  mkdir -p ${package_name}-${version}/SRPMS
  mkdir -p ${package_name}-${version}/SOURCES

  local sha256_x86_64
  curl -L ${url_x86_64} -o ${package_name}-${version}/SOURCES/${package_name}-${version}-x86_64.tar.gz
  sha256_x86_64=$(sha256sum ${package_name}-${version}/SOURCES/${package_name}-${version}-x86_64.tar.gz | awk '{ print $1 }')

  local sha256_aarch64
  curl -L ${url_aarch64} -o ${package_name}-${version}/SOURCES/${package_name}-${version}-aarch64.tar.gz
  sha256_aarch64=$(sha256sum ${package_name}-${version}/SOURCES/${package_name}-${version}-aarch64.tar.gz | awk '{ print $1 }')

  # 创建通用 SPEC 文件
  cat <<EOF > ${package_name}-${version}/SPECS/${package_name}.spec
%define name ${package_name}
%define version ${version}

Name: %{name}
Version: %{version}
Release: 1%{?dist}
Summary: ${DESCRIPTION}

License: ${LICENSE}
URL: ${REPO_URL}

%description
${LONG_DESCRIPTION}

%ifarch x86_64
Source0: %{name}-%{version}-x86_64.tar.gz
%endif
%ifarch aarch64
Source0: %{name}-%{version}-aarch64.tar.gz
%endif

%prep
%ifarch x86_64
%define expected_sha256sum ${sha256_x86_64}
%endif
%ifarch aarch64
%define expected_sha256sum ${sha256_aarch64}
%endif

echo "%{expected_sha256sum}  %{_sourcedir}/%{SOURCE0}" | sha256sum -c -
%setup -q -n %{name}-%{version}-%{_target_cpu}

%build

%install
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/include
mkdir -p %{buildroot}/usr/lib
cp -r * %{buildroot}/usr/

%files
/usr/bin/*
/usr/include/*
/usr/lib/*
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

# Create Chocolatey Package
create_choco_package() {
  echo "Creating Chocolatey Package..."

  local package_name=$1
  local version=${2#v}
  local url_x86_64=$3
  local url_x86=$4
  local url_arm64=$5

  # Extract the name part from MAINTAINER
  MAINTAINER_NAME=$(echo $MAINTAINER | sed 's/ <.*//')

  # 创建临时目录
  mkdir -p ${package_name}-${version}/tools

  # 计算 SHA-256 校验和
  local sha256_x86_64
  sha256_x86_64=$(curl -L ${url_x86_64} | sha256sum | awk '{ print $1 }')

  local sha256_x86
  sha256_x86=$(curl -L ${url_x86} | sha256sum | awk '{ print $1 }')

  local sha256_arm64
  sha256_arm64=$(curl -L ${url_arm64} | sha256sum | awk '{ print $1 }')

  # 创建 .nuspec 文件
  cat <<EOF > ${package_name}-${version}/${package_name}.nuspec
<?xml version="1.0"?>
<package xmlns="http://schemas.microsoft.com/packaging/2011/08/nuspec.xsd">
  <metadata>
    <id>${package_name}</id>
    <version>${version}</version>
    <authors>${MAINTAINER_NAME}</authors>
    <owners>${MAINTAINER_NAME}</owners>
    <description>${DESCRIPTION}</description>
    <licenseUrl>${LICENSE_URL}</licenseUrl>
    <projectUrl>${REPO_URL}</projectUrl>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
  </metadata>
  <files>
    <file src="tools\chocolateyInstall.ps1" target="tools\chocolateyInstall.ps1" />
    <file src="tools\chocolateyUninstall.ps1" target="tools\chocolateyUninstall.ps1" />
  </files>
</package>
EOF

  # 创建 install.ps1 脚本
  cat <<EOF > ${package_name}-${version}/tools/chocolateyInstall.ps1
# Download and install the software
\$ErrorActionPreference = 'Stop'

\$packageName = '${package_name}'
\$checksumType = 'sha256'

if (\$env:PROCESSOR_ARCHITECTURE -eq 'AMD64') {
  \$url = '${url_x86_64}'
  \$checksum = '${sha256_x86_64}'
} elseif (\$env:PROCESSOR_ARCHITECTURE -eq 'x86') {
  \$url = '${url_x86}'
  \$checksum = '${sha256_x86}'
} elseif (\$env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
  \$url = '${url_arm64}'
  \$checksum = '${sha256_arm64}'
} else {
  throw "Unsupported architecture: \$env:PROCESSOR_ARCHITECTURE"
}

Install-ChocolateyPackage "\$packageName" 'exe' "\$url" --checksum "\$checksum" --checksumType "\$checksumType"
EOF

  # 创建 uninstall.ps1 脚本
  cat <<EOF > ${package_name}-${version}/tools/chocolateyUninstall.ps1
# Uninstall the software
\$ErrorActionPreference = 'Stop'
\$packageName = '${package_name}'
Remove-Item -Recurse -Force "\$env:ChocolateyInstall\lib\$packageName"
EOF

  # 创建 CHANGELOG.md 文件
  cat <<EOF > ${package_name}-${version}/CHANGELOG.md
# Changelog

## ${version}
- See CHANGELOG.md in package
EOF

  # 打包 choco 包，获取 ./${package_name}-${version}.nupkg
  choco pack ${package_name}-${version}/${package_name}.nuspec

  # 清理临时目录
  rm -rf ${package_name}-${version}

  echo "Chocolatey packages made: ${package_name}-${version}.nupkg"
  echo "Chocolatey package created successfully"
}

package() {
    OS=$(uname -s)

    case "$OS" in
        Linux*)
            echo "==========================================================="
            create_rpm_package "adi" "${VERSION}" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-x86_64.tar.gz" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-aarch64.tar.gz"

            echo "==========================================================="
            create_debian_package "adi" "${VERSION}" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-x86_64.tar.gz" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-linux-aarch64.tar.gz"
            echo "==========================================================="
            ;;

        Darwin*)
            echo "==========================================================="
            create_homebrew_formula "adi" "${VERSION}" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-macos-x86_64.tar.gz" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-macos-arm64.tar.gz"
            echo "==========================================================="
            ;;

        CYGWIN*|MINGW*|MSYS*)
            echo "==========================================================="
            create_choco_package "adi" "${VERSION}" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-windows-x86_64.zip" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-windows-x86.zip" \
              "https://github.com/Windsander/ADI-Stable-Diffusion/releases/download/release-${VERSION}/release-${VERSION}-windows-arm64.zip"
            echo "==========================================================="
            ;;

        *)
            echo "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
}

ensure_tools
package

echo "Deployment completed successfully."