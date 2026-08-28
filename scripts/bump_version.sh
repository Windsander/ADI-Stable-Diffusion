#!/bin/bash
#
# bump_version.sh — bump ADI to a new version.
#
# include/adi.h (CURRENT_ADI_VERSION) is the SINGLE SOURCE OF TRUTH for the
# ADI version. This script updates it and scaffolds the CHANGELOG section.
# auto-publish.yml refuses to cut a release whose branch name disagrees with
# the header, so versions can no longer drift apart.
#
# Usage:
#   scripts/bump_version.sh v2.1.0
#
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 v<major>.<minor>.<patch>   (e.g. v2.1.0)"
    exit 1
fi

NEW_VERSION="$1"
if ! [[ "${NEW_VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: version must look like vX.Y.Z, got '${NEW_VERSION}'"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEADER="${REPO_ROOT}/include/adi.h"
CHANGELOG="${REPO_ROOT}/CHANGELOG.md"

OLD_VERSION=$(grep -oE '#define CURRENT_ADI_VERSION "v[0-9]+\.[0-9]+\.[0-9]+"' "${HEADER}" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
if [ -z "${OLD_VERSION}" ]; then
    echo "ERROR: CURRENT_ADI_VERSION not found in ${HEADER}"
    exit 1
fi
if [ "${OLD_VERSION}" = "${NEW_VERSION}" ]; then
    echo "Already at ${NEW_VERSION}, nothing to do."
    exit 0
fi
if grep -q "## \[${NEW_VERSION}\]" "${CHANGELOG}"; then
    echo "ERROR: CHANGELOG.md already has a section for ${NEW_VERSION}"
    exit 1
fi

echo "Bumping ${OLD_VERSION} -> ${NEW_VERSION}"

# 1. SSOT: include/adi.h (portable in-place edit, no GNU/BSD sed split)
sed -i.bak "s/#define CURRENT_ADI_VERSION \"${OLD_VERSION}\"/#define CURRENT_ADI_VERSION \"${NEW_VERSION}\"/" "${HEADER}"
rm -f "${HEADER}.bak"
grep -n "CURRENT_ADI_VERSION" "${HEADER}"

# 2. CHANGELOG: scaffold a new section right below the '# Changelog' title
TODAY=$(date +%Y-%m-%d)
awk -v ver="${NEW_VERSION}" -v day="${TODAY}" '
    BEGIN { done = 0 }
    /^# Changelog/ && done == 0 {
        print
        print ""
        print "## [" ver "] - " day
        print ""
        print "### Added"
        print "- TODO"
        print ""
        print "### Changed"
        print "- TODO"
        print ""
        print "### Fixed"
        print "- TODO"
        done = 1
        next
    }
    { print }
' "${CHANGELOG}" > "${CHANGELOG}.tmp"
mv "${CHANGELOG}.tmp" "${CHANGELOG}"

echo ""
echo "Done. Next steps:"
echo "  1. Fill in the ${NEW_VERSION} section in CHANGELOG.md"
echo "  2. git add include/adi.h CHANGELOG.md && git commit -m \"chore: bump version to ${NEW_VERSION}\""
echo "  3. git checkout -b release/release-${NEW_VERSION} && git push -u origin release/release-${NEW_VERSION}"
echo "  4. auto-publish will verify branch version == CURRENT_ADI_VERSION and cut the release"
