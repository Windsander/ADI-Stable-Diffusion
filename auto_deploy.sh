#!/bin/bash

set -e

# Variables from environment
VERSION="${VERSION}"
TARBALL_LOG="${TARBALL_LOG}"
TARBALL_URL="${TARBALL_URL}"
TARBALL_SHA256="${TARBALL_SHA256}"

# Static variables
REPO_URL="https://github.com/Windsander/ADI-Stable-Diffusion"
DESCRIPTION="Agile Diffusers Inference (ADI) Command Line Tool"
LONG_DESCRIPTION="Agile Diffusers Inference (ADI) is a C++ project. Its purpose is to leverage the acceleration capabilities of ONNXRuntime and the high compatibility of the .onnx model format to provide a convenient solution for the engineering deployment of Stable Diffusion."
MAINTAINER="Arikan.Li<https://github.com/Windsander/ADI-Stable-Diffusion/issues>"
LICENSE="GPL-3.0 license"
RELEASE="1%{?dist}"

# Ensure necessary tools are installed
ensure_tools() {
    echo "Ensuring necessary tools are installed..."
    if ! command -v cmake &> /dev/null; then
        echo "CMake not found, installing..."
        sudo apt-get install -y cmake || sudo yum install -y cmake || brew install cmake
    fi
    if ! command -v ninja &> /dev/null; then
        echo "Ninja not found, installing..."
        sudo apt-get install -y ninja-build || sudo yum install -y ninja || brew install ninja
    fi
    if ! command -v dput &> /dev/null; then
        echo "dput not found, installing..."
        sudo apt-get install -y dput
    fi
    if ! command -v copr-cli &> /dev/null; then
        echo "copr-cli not found, installing..."
        sudo dnf install -y copr-cli || sudo yum install -y copr-cli
    fi
}

# Read changelog from TARBALL_LOG
read_changelog() {
    echo "Reading changelog from $TARBALL_LOG..."
    CHANGELOG=$(cat "$TARBALL_LOG")
}

# Create Homebrew Formula
create_homebrew_formula() {
    echo "Creating Homebrew Formula..."
    mkdir -p Formula
    cat <<EOF > Formula/adi.rb
class Adi < Formula
  desc "$DESCRIPTION"
  homepage "$REPO_URL"
  url "$TARBALL_URL"
  sha256 "$TARBALL_SHA256"
  license "$LICENSE"

  def install
    bin.install "bin/ort-sd-clitools"
    lib.install Dir["lib/*"]
    include.install Dir["include/*"]
  end

  test do
    system "#{bin}/ort-sd-clitools", "--version"
  end
end
EOF
}

# Create Debian Package
create_debian_package() {
    echo "Creating Debian Package..."
    mkdir -p adi-$VERSION
    cd adi-$VERSION
    wget $TARBALL_URL -O v$VERSION.tar.gz
    tar -xzf v$VERSION.tar.gz --strip-components=1
    mkdir -p debian
    cd debian
    dh_make -s -y -c $LICENSE -p adi_$VERSION
    cat <<EOF > control
Source: adi
Section: utils
Priority: optional
Maintainer: $MAINTAINER
Build-Depends: debhelper (>= 9)
Standards-Version: 3.9.6
Homepage: $REPO_URL

Package: adi
Architecture: any
Depends: \${shlibs:Depends}, \${misc:Depends}
Description: $DESCRIPTION
 $LONG_DESCRIPTION
EOF
    cd ..
    cat <<EOF > debian/rules
#!/usr/bin/make -f

%:
	dh \$@

override_dh_auto_install:
	# Install the executable
	install -m 755 bin/ort-sd-clitools \$(DESTDIR)/usr/bin/ort-sd-clitools
	# Install the library files
	install -d \$(DESTDIR)/usr/lib
	install -m 644 lib/* \$(DESTDIR)/usr/lib/
	# Install the header files
	install -d \$(DESTDIR)/usr/include
	install -m 644 include/* \$(DESTDIR)/usr/include/
EOF
    chmod +x debian/rules
    cat <<EOF > debian/changelog
adi ($VERSION-1) unstable; urgency=low

  * $DESCRIPTION

$CHANGELOG

 -- $MAINTAINER  $(date -R)
EOF
    debuild -us -uc
    cd ..
    mkdir -p ../debian
    mv ../adi_$VERSION-1_amd64.deb ../debian/
}

# Create RPM Package
create_rpm_package() {
    echo "Creating RPM Package..."
    mkdir -p ~/rpmbuild/SOURCES
    wget $TARBALL_URL -O ~/rpmbuild/SOURCES/adi-$VERSION.tar.gz
    cat <<EOF > ~/rpmbuild/SPECS/adi.spec
Name:           adi
Version:        $VERSION
Release:        $RELEASE
Summary:        $DESCRIPTION

License:        $LICENSE
URL:            $REPO_URL
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  cmake
Requires:       cmake

%description
$LONG_DESCRIPTION

%prep
%setup -q

%build
# If there are any build steps needed, they can be added here, currently seems unnecessary

%install
rm -rf %{buildroot}
# Install the executable
install -m 755 bin/ort-sd-clitools %{buildroot}%{_bindir}/ort-sd-clitools
# Install the library files
install -d %{buildroot}%{_libdir}
install -m 644 lib/* %{buildroot}%{_libdir}/
# Install the header files
install -d %{buildroot}%{_includedir}
install -m 644 include/* %{buildroot}%{_includedir}/

%files
%license LICENSE
%doc README.md
%{_bindir}/ort-sd-clitools
%{_libdir}/*
%{_includedir}/*

%changelog
* $(date "+%a %b %d %Y") $MAINTAINER - $VERSION-1
- $DESCRIPTION

$CHANGELOG
EOF

    rpmbuild -ba ~/rpmbuild/SPECS/adi.spec
    mkdir -p ../rpms/x86_64
    mv ~/rpmbuild/RPMS/x86_64/adi-$VERSION-1.x86_64.rpm ../rpms/x86_64/
}

# Main function
main() {
    ensure_tools
    read_changelog
    create_homebrew_formula
    create_debian_package
    create_rpm_package
}

main