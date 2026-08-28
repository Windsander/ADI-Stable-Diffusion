#!/usr/bin/env bash
# ADI 主站一键部署：本地构建 → 推送 gh-pages 分支
# 用法：bash scripts/deploy-gh-pages.sh
set -euo pipefail

SITE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
GIT_BIN="${GIT_BIN:-/usr/local/bin/git}"
REMOTE="${REMOTE:-git@github.com:Windsander/ADI-Stable-Diffusion.git}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$SITE_DIR"
npm run build

cp -R dist/. "$TMP"/
cd "$TMP"
"$GIT_BIN" init -q
"$GIT_BIN" symbolic-ref HEAD refs/heads/gh-pages
touch .nojekyll
"$GIT_BIN" add -A
"$GIT_BIN" -c user.name=Windsander -c user.email=Windsander@users.noreply.github.com \
  commit -q -m "ADI 主站静态构建 · $(date '+%Y-%m-%d %H:%M')"
"$GIT_BIN" push --force "$REMOTE" gh-pages
echo "✅ 已推送 gh-pages，Pages 将在约 1 分钟内更新 https://adi.cyberfederal.io"
