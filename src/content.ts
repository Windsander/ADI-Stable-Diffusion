// ============================================================
// ADI 站点内容数据 —— 全部来自 ADI-Stable-Diffusion 仓库真实记录
// ============================================================

export const REPO_URL = "https://github.com/Windsander/ADI-Stable-Diffusion";
export const RELEASES_URL = `${REPO_URL}/releases`;
export const ROADMAP_URL = `${REPO_URL}/blob/main/ROADMAP.md`;
export const CHANGELOG_URL = `${REPO_URL}/blob/main/CHANGELOG.md`;

export const stats = [
  { value: "9", unit: "平台产物", note: "Android×4 · Linux×2 · macOS · Windows×2" },
  { value: "14+", unit: "采样器", note: "离散全家桶 + Karras σ + flow 家族" },
  { value: "6", unit: "模型家族", note: "SD1.x → SD3.5 / FLUX / SVD" },
  { value: "3", unit: "分词器", note: "BPE · WordPiece · SentencePiece" },
  { value: "1e-7", unit: "轨迹偏差量级", note: "与 diffusers 交叉验证 1e-5 ~ 1e-7" },
];

export interface ShowcaseItem {
  model: string;
  media: string;
  kind: "image" | "video";
  resolution: string;
  steps: string;
  extra: string;
}

export const showcaseItems: ShowcaseItem[] = [
  { model: "FLUX.1-schnell", media: "images/showcase-flux.jpg", kind: "image", resolution: "1024×1024", steps: "4 steps", extra: "flow_euler · MMDiT" },
  { model: "SD3.5-turbo", media: "images/showcase-sd35.jpg", kind: "image", resolution: "1024×1024", steps: "4 steps", extra: "T5-XXL · 三路编码" },
  { model: "SD v2.1", media: "images/showcase-sd21.jpg", kind: "image", resolution: "768×768", steps: "20 steps", extra: "v_prediction" },
  { model: "SDXL-turbo", media: "images/showcase-sdxl.jpg", kind: "image", resolution: "512×512", steps: "4 steps", extra: "双文本编码器" },
  { model: "sd-turbo", media: "images/showcase-sdturbo.jpg", kind: "image", resolution: "512×512", steps: "4 steps", extra: "euler_a · ≈9.7s" },
  { model: "SVD img2vid", media: "videos/svd-img2vid.mp4", kind: "video", resolution: "14 帧", steps: "25 steps", extra: "euler_svd · 单图驱动" },
];

export interface Feature {
  tag: string;
  title: string;
  body: string;
}

export const features: Feature[] = [
  {
    tag: "PURE C++17",
    title: "零 Python 推理时",
    body: "CLIP / UNet / VAE 由纯 C++17 编排。公共面只有 include/adi.h 一个头文件、六个 C ABI 入口，任何能链接 C ABI 的宿主都能直接嵌入。",
  },
  {
    tag: "EDM σ-SPACE",
    title: "一个坐标系的采样器",
    body: "全部采样器统一在 EDM 约定 x = x0 + σ·ε 中运作；Karras σ 策略（ρ=7）在唯一注入点落地，可与任意采样器组合。",
  },
  {
    tag: "MMDiT READY",
    title: "进入整流场时代",
    body: "SD3.5 三路文本编码（含 SentencePiece 驱动的 T5-XXL）、FLUX packed latents 与 rotary ids、flow_euler 调度——全部在 CLI 参数面上暴露。",
  },
  {
    tag: "IMG2VID",
    title: "照片长成影像",
    body: "SVD 模式以 CLIP 视觉编码器 + 时空 UNet + euler_svd 采样器，把单张输入照片展开为 14 帧连续画面，逐帧 PNG 输出。",
  },
  {
    tag: "PRECISION AUTO",
    title: "低内存也跑 1024px",
    body: "--precision auto|fp32|fp16：探测可用内存，按需派生 fp16 模型副本并缓存复用，受限机器不必手工转换模型。",
  },
  {
    tag: "RELEASE CHAIN",
    title: "一次提交九份产物",
    body: "自动化发布链随 release 产出九个平台包；macOS 走 Homebrew、Windows 走 Chocolatey，也可 auto_build.sh 一键交叉编译。",
  },
];

export interface PerfRow {
  model: string;
  resolution: string;
  steps: string;
  time: string;
}

export const perfRows: PerfRow[] = [
  { model: "sd-turbo", resolution: "512×512", steps: "4", time: "≈ 9.7 s" },
  { model: "SD3.5-turbo（MMDiT · 三路编码器）", resolution: "1024×1024", steps: "4", time: "≈ 164 s" },
];

export const perfNote = "Apple M4 Max（128GB）· ONNXRuntime 1.28.0 · 默认 Provider · 单次冷启动 · 无 GPU · 无 Python";

export interface InstallMethod {
  id: string;
  label: string;
  lines: string[];
}

export const installMethods: InstallMethod[] = [
  {
    id: "brew",
    label: "macOS · Homebrew",
    lines: ["brew tap windsander/adi-stable-diffusion", "brew install adi"],
  },
  {
    id: "choco",
    label: "Windows · Chocolatey",
    lines: [
      'curl -L -o adi.1.0.1.nupkg "https://raw.githubusercontent.com/Windsander/ADI-Stable-Diffusion/deploy/adi.1.0.1.nupkg"',
      "choco install adi.1.0.1.nupkg -y",
    ],
  },
  {
    id: "source",
    label: "源码构建",
    lines: [
      "git clone https://github.com/Windsander/ADI-Stable-Diffusion.git",
      "bash ./auto_build.sh --platform macos --build-type release",
    ],
  },
];

export interface MatrixRow {
  model: string;
  dims: string;
  predictor: string;
  decoding: string;
  note: string;
}

export const matrixRows: MatrixRow[] = [
  { model: "SD v1.x / turbo", dims: "768 / 1024", predictor: "epsilon", decoding: "0.18215", note: "turbo：guidance 1.0，1~4 步" },
  { model: "SD v2.1", dims: "1024", predictor: "v_prediction", decoding: "0.18215", note: "v2.1-768 用 768px" },
  { model: "SDXL / SDXL-turbo", dims: "768", predictor: "epsilon", decoding: "0.13025", note: "需要 --clip2" },
  { model: "SD3.5 / SD3.5-turbo", dims: "768", predictor: "epsilon", decoding: "1.5305", note: "--clip2 + --clip3 + --sp-model；flow_euler，shift 3.0" },
  { model: "FLUX.1-schnell", dims: "768", predictor: "epsilon", decoding: "0.3611", note: "flow_euler，shift 1.0，guidance 1.0" },
  { model: "SVD（img2vid）", dims: "768", predictor: "v_prediction", decoding: "0.18215", note: "euler_svd；--frames / --fps / --motion-bucket" },
];

export const schedulerList = [
  "euler", "euler_a", "lms", "lcm", "heun", "ddpm", "ddim",
  "unipc", "dpm_m", "dpm_sde", "dpm_s", "pndm", "ipndm", "deis_m",
  "flow_euler", "euler_svd",
];

export const footerColumns = [
  {
    heading: "项目",
    links: [
      { text: "GitHub 仓库", href: REPO_URL },
      { text: "Releases", href: RELEASES_URL },
      { text: "Roadmap", href: ROADMAP_URL },
      { text: "Changelog", href: CHANGELOG_URL },
    ],
  },
  {
    heading: "生态",
    links: [
      { text: "ONNXRuntime", href: "https://onnxruntime.ai" },
      { text: "HF diffusers", href: "https://huggingface.co/docs/diffusers" },
      { text: "optimum", href: "https://huggingface.co/docs/optimum" },
    ],
  },
  {
    heading: "作者",
    links: [
      { text: "Windsander", href: "https://github.com/Windsander" },
      { text: "arikanli.cyberfederal.io", href: "https://arikanli.cyberfederal.io" },
    ],
  },
];
