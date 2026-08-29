#!/usr/bin/env bash
# 导入项目助手所需的 GitHub Labels
# 用法: bash scripts/import-labels.sh [owner/repo]
#   未提供参数时，依次尝试：git remote origin → 默认仓库
# 要求: gh CLI 已认证 (gh auth status)，且安装了 jq
#
# 幂等：gh label create --force 会在标签已存在时直接更新颜色与描述。

set -euo pipefail

DEFAULT_REPO="Windsander/ADI-Stable-Diffusion"
LABELS_FILE="$(cd "$(dirname "$0")" && pwd)/github-labels.json"

if ! command -v gh &> /dev/null; then
  echo "错误: 需要 GitHub CLI (gh)。请先安装: https://cli.github.com/"
  exit 1
fi

if ! command -v jq &> /dev/null; then
  echo "错误: 需要 jq 解析 $LABELS_FILE。请先安装（如 brew install jq）。"
  exit 1
fi

if [ ! -f "$LABELS_FILE" ]; then
  echo "错误: 未找到 $LABELS_FILE"
  exit 1
fi

# 仓库解析优先级：命令行参数 > git remote origin > 默认值
REPO="${1:-}"
if [ -z "$REPO" ]; then
  remote_url="$(git remote get-url origin 2>/dev/null || true)"
  if [[ "$remote_url" =~ github\.com[:/]([^/]+/[^/.]+)(\.git)?$ ]]; then
    REPO="${BASH_REMATCH[1]}"
  else
    REPO="$DEFAULT_REPO"
  fi
fi

echo "正在为 $REPO 创建/更新 Labels..."

fail_count=0
while read -r label; do
  name="$(jq -r '.name' <<< "$label")"
  color="$(jq -r '.color' <<< "$label")"
  description="$(jq -r '.description' <<< "$label")"

  # --force：已存在则更新，不存在则创建
  if gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force; then
    echo "  ✓ $name"
  else
    echo "  ✗ $name 创建/更新失败" >&2
    fail_count=$((fail_count + 1))
  fi
done < <(jq -c '.[]' "$LABELS_FILE")

if [ "$fail_count" -gt 0 ]; then
  echo "完成，但有 $fail_count 个标签失败，请检查 gh 认证与仓库权限。"
  exit 1
fi
echo "Labels 导入完成（共 $(jq length "$LABELS_FILE") 个）。"
