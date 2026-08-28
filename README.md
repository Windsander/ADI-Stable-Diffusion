# ADI-site

**[ADI-Stable-Diffusion](https://github.com/Windsander/ADI-Stable-Diffusion)（Agile Diffusers Inference）的项目主站** —— 托管于 GitHub Pages，线上地址：

**https://adi.cyberfederal.io**

工程产品风单页站：点阵网格 Hero + 真实 CLI 终端卡片、九平台/采样器等关键数据条、六格实测展厅（5 张本机推理原图 + SVD img2vid 循环视频）、特性卡片、M4 Max CPU 实测性能表与采样器清单、三种安装方式、SD1.x → FLUX / SVD 参数矩阵表。站内全部展示素材均为 ADI 引擎本机推理的真实输出。

## 技术栈

- React 19 + TypeScript + Vite 7 + Tailwind CSS 3.4
- 字体：Inter / Noto Sans SC（正文）· JetBrains Mono（参数与命令）

## 本地开发

```bash
npm install
npm run dev        # 开发服务器
npm run build      # 产物在 dist/，postbuild 自动复制 404.html（SPA 回退）
```

## 内容维护

所有文案与数据集中在 `src/content.ts`；展厅图在 `public/images/`，SVD 循环视频在 `public/videos/svd-img2vid.mp4`。

## 部署

push 到 `main` 即由 `.github/workflows/deploy-pages.yml` 自动构建并发布到 GitHub Pages。域名通过 `public/CNAME`（`adi.cyberfederal.io`）绑定，DNS 侧将 `adi` 子域 CNAME 指向 `windsander.github.io` 即可。
