# 深度学习图像生成算法调研

> 面向 `gentask`（在 `segtask` 基础上扩展的生成任务）的算法选型与背景调研文档。
>
> 目标：按**发展脉络**梳理图像生成领域中公认经典、经过时间检验、对后续工作有关键影响的深度学习算法。每个算法用一段话给出**最关键信息**（核心思想 / 贡献 / 局限），并附**官方 GitHub 实现**；若无官方实现，则附经过社区检验的高质量第三方实现。
>
> 链接均为撰写时（2026-06）可访问的仓库地址。第三方实现以「⚙️ 第三方」标注，官方实现以「🏠 官方」标注。

---

## 0. 阅读指南：五大生成范式

现代深度生成模型大致可归为五条技术路线，理解它们的取舍有助于选型：

| 范式 | 核心思想 | 代表 | 采样质量 | 似然/可控性 | 训练稳定性 | 采样速度 |
| --- | --- | --- | --- | --- | --- | --- |
| 自回归 (AR) | 逐像素/逐 token 条件概率分解 | PixelCNN、ImageGPT、VQGAN+Transformer | 高 | 精确似然 | 稳定 | 慢（逐元素） |
| VAE | 变分下界 + 编解码隐空间 | VAE、NVAE、VQ-VAE | 中（偏糊） | 隐变量可控 | 稳定 | 快 |
| GAN | 判别器对抗博弈 | DCGAN、StyleGAN、BigGAN | 很高 | 无显式似然 | 易崩溃/模式坍塌 | 极快（单步） |
| 归一化流 (Flow) | 可逆变换 + 雅可比行列式 | RealNVP、Glow | 中 | 精确似然 | 稳定 | 快 |
| 扩散/分数模型 | 逐步去噪 / 分数匹配 | DDPM、LDM/SD、EDM | SOTA | 高可控（引导） | 非常稳定 | 慢（多步，但可蒸馏加速） |

> 选型直觉：**追求最高质量 + 强可控性** → 扩散模型；**追求实时单步生成** → GAN / 一致性模型；**需要精确似然** → AR / Flow；**做表示学习或离散隐空间** → VAE / VQ-VAE。
>
> 与本项目相关的三类任务（超分、去噪、Inpainting）在第 7 节单列。

---

## 1. 自回归模型（Autoregressive, AR）

将图像建模为像素（或离散 token）序列，用链式法则把联合分布分解为条件分布的乘积 `p(x)=∏ p(x_i | x_<i)`。优点是似然精确、训练稳定；缺点是采样需逐元素串行，速度慢。

- **PixelRNN / PixelCNN (2016)** — 自回归图像生成的奠基工作。用带掩码卷积（masked conv）保证「只看已生成像素」的因果性，PixelRNN 用二维 LSTM、PixelCNN 用卷积更快。确立了「逐像素建模 + 离散 softmax 输出」范式。
  - ⚙️ 第三方（含 PixelCNN++，作者 OpenAI）：https://github.com/openai/pixel-cnn

- **PixelCNN++ (2017)** — 对 PixelCNN 的工程化改进：用离散化 logistic 混合似然替代 256 路 softmax、下采样+短路连接、单通道条件化，显著提升似然与速度。是后续 AR/扩散输出头设计的重要参考。
  - 🏠 官方：https://github.com/openai/pixel-cnn

- **VQ-VAE (2017) / VQ-VAE-2 (2019)** — 用「向量量化」把图像编码成**离散** latent code（codebook），再在离散隐空间上训练自回归先验（PixelCNN/Transformer）。VQ-VAE-2 引入多尺度分层量化，首次让 AR 路线生成高保真大图。是「先压缩成 token、再自回归建模」两阶段范式的源头，深刻影响了 VQGAN、DALL·E、潜空间扩散。
  - 🏠 官方（DeepMind Sonnet 内含 VQ-VAE 实现）：https://github.com/deepmind/sonnet
  - ⚙️ 第三方（VQ-VAE-2 PyTorch，高星）：https://github.com/rosinality/vq-vae-2-pytorch

- **Image Transformer (2018)** — 把自注意力引入像素级自回归生成，用局部自注意力缓解长序列开销，是 Transformer 进入图像生成的早期标志。
  - ⚙️ 第三方（在 tensor2tensor 中）：https://github.com/tensorflow/tensor2tensor

- **VQGAN / Taming Transformers (2021)** — VQ-VAE 的「感知增强」版：量化阶段加入 GAN 判别器 + 感知损失，得到信息密度高、视觉锐利的 codebook；再用 Transformer 在 token 上做自回归。奠定了「VQGAN 压缩 + Transformer 先验」的高分辨率生成范式，也是 Stable Diffusion 自编码器的直接前身。
  - 🏠 官方：https://github.com/CompVis/taming-transformers

- **ImageGPT (2020)** — 直接把 GPT 套到像素序列上，证明「纯自回归 Transformer + 大数据」可学到强图像表示与生成能力，是「生成式预训练做视觉表示学习」的代表。
  - 🏠 官方：https://github.com/openai/image-gpt

- **MaskGIT (2022)** — 用**双向**掩码 Transformer 做并行迭代解码（一次预测多个 token、按置信度逐轮填充），把 AR 的串行采样从 O(N) 降到常数轮数，质量/速度兼顾。是 Muse、Parti 等掩码生成路线的基础。
  - 🏠 官方：https://github.com/google-research/maskgit

---

## 2. 变分自编码器（VAE）

通过编码器近似后验、解码器重建，最大化证据下界（ELBO）。训练稳定、隐空间连续可插值，但重建偏模糊。是表示学习与隐空间生成的基石。

- **VAE (2013, Kingma & Welling)** — 提出重参数化技巧（reparameterization trick）使变分推断可端到端反向传播，奠定了「编码器-解码器 + KL 正则」的概率生成框架。
  - ⚙️ 第三方（PyTorch 官方示例）：https://github.com/pytorch/examples/tree/main/vae

- **β-VAE (2017)** — 在 KL 项加权重 β（>1）以鼓励隐变量解耦（disentanglement），是无监督可解释表示研究的代表。
  - ⚙️ 第三方（高星实现）：https://github.com/1Konny/Beta-VAE

- **NVAE (2020)** — 深层分层 VAE，靠精心设计的残差/深度可分离结构与谱正则稳定训练，把 VAE 的生成质量推到接近 GAN，证明 VAE 路线的上限。
  - 🏠 官方：https://github.com/NVlabs/NVAE

- **VDVAE (2020)** — "Very Deep VAE"，证明足够深的分层 VAE 在对数似然上可超过自回归模型，结构简洁。
  - 🏠 官方：https://github.com/openai/vdvae

> 注：VQ-VAE 虽名为 VAE，但其离散量化 + 自回归先验的用法更接近 AR 范式，已在第 1 节列出。

---

## 3. 生成对抗网络（GAN）

生成器与判别器对抗博弈，无需显式似然即可生成极锐利的样本。曾是 2014–2021 年图像生成的主流，至今在实时单步生成、超分、图像翻译中仍是强基线。主要难点是训练不稳定与模式坍塌。

- **GAN (2014, Goodfellow)** — 提出对抗训练框架（min-max 博弈），开创无显式似然的隐式生成范式。
  - ⚙️ 第三方（作者后续 Improved GAN，含评测代码）：https://github.com/openai/improved-gan

- **DCGAN (2015)** — 第一个稳定可用的全卷积 GAN 架构（转置卷积、BatchNorm、无全连接、LeakyReLU），并展示隐空间语义算术，是 GAN 工程化的起点。
  - 🏠 官方（Theano）：https://github.com/Newmu/dcgan_code
  - ⚙️ 第三方（PyTorch 官方示例）：https://github.com/pytorch/examples/tree/main/dcgan

- **WGAN (2017) / WGAN-GP (2017)** — 用 Wasserstein 距离替代 JS 散度缓解梯度消失，WGAN-GP 用梯度惩罚替代权重裁剪，大幅提升训练稳定性，是 GAN 损失理论的里程碑。
  - 🏠 官方（WGAN）：https://github.com/martinarjovsky/WassersteinGAN
  - 🏠 官方（WGAN-GP）：https://github.com/igul222/improved_wgan_training

- **cGAN / Pix2Pix (2017)** — 条件 GAN 用于成对图像翻译（label→image、边缘→照片），提出 PatchGAN 判别器 + L1 重建。是图像到图像翻译（包括超分/去噪/上色）的通用骨架。
  - 🏠 官方（与 CycleGAN 同仓）：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- **CycleGAN (2017)** — 引入循环一致性损失，实现**无需成对数据**的域翻译（马↔斑马、夏↔冬），是非配对图像翻译的代表。
  - 🏠 官方：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- **ProGAN / PGGAN (2018)** — 渐进式增长：从低分辨率开始逐级加层训练到 1024²，首次稳定生成高清人脸，并提出 CelebA-HQ。
  - 🏠 官方：https://github.com/tkarras/progressive_growing_of_gans

- **SAGAN (2018)** — 把自注意力引入 GAN 以建模长程依赖，并引入谱归一化 + TTUR，提升 ImageNet 类条件生成的全局一致性。
  - 🏠 官方：https://github.com/brain-research/self-attention-gan

- **BigGAN (2019)** — 大 batch、大模型、截断技巧（truncation trick）+ 正交正则，把类条件 ImageNet 生成的 FID 推到当时 SOTA，验证「规模化」对 GAN 的作用。
  - 🏠 官方（作者 PyTorch 版）：https://github.com/ajbrock/BigGAN-PyTorch

- **StyleGAN 系列 (2019–2021)** — 生成质量与可控性的标杆。
  - **StyleGAN (v1)**：映射网络 + AdaIN 风格注入 + 多尺度噪声，实现高质量且可分层控制的人脸生成。🏠 https://github.com/NVlabs/stylegan
  - **StyleGAN2**：去除水滴伪影（weight demodulation）、路径长度正则，质量再上台阶。🏠 https://github.com/NVlabs/stylegan2
  - **StyleGAN2-ADA**：自适应判别器数据增强，小数据集也能稳定训练。🏠 https://github.com/NVlabs/stylegan2-ada-pytorch
  - **StyleGAN3**：消除「纹理粘连」实现真正的平移/旋转等变，利于视频与动画。🏠 https://github.com/NVlabs/stylegan3

- **StyleGAN-XL (2022)** — 把 StyleGAN3 扩展到 ImageNet 类大规模多类别数据并冲到 SOTA FID，是 GAN 路线在扩散浪潮下的有力回应。
  - 🏠 官方：https://github.com/autonomousvision/stylegan-xl

---

## 4. 归一化流（Normalizing Flows）

用一系列**可逆**变换把简单分布（高斯）映射到数据分布，通过变量替换公式精确计算似然。优点是精确似然 + 快速并行采样，缺点是可逆性约束限制表达力、模型偏大。

- **NICE (2014) / RealNVP (2017)** — 提出耦合层（coupling layer）使雅可比行列式易计算，RealNVP 用仿射耦合 + 多尺度结构，是现代归一化流的基础。
  - ⚙️ 第三方（RealNVP PyTorch）：https://github.com/chrischute/real-nvp

- **Glow (2018)** — 引入可逆 1×1 卷积替代固定置换，显著提升流模型的生成质量并展示隐空间属性插值，是 Flow 路线最具影响力的工作。
  - 🏠 官方：https://github.com/openai/glow

---

## 5. 扩散 / 分数模型（Diffusion & Score-based）

通过「前向逐步加噪 → 反向逐步去噪」学习数据分布，等价于分数（score, ∇ log p）匹配。2020 年后成为图像生成的主导范式：训练极稳定、质量 SOTA、可控性强（条件/引导），唯一短板是多步采样慢（已被蒸馏/少步采样大幅缓解）。

- **DDPM (2020)** — 把扩散概率模型与去噪目标统一，提出简化的噪声预测损失 `‖ε − ε_θ‖²`，首次让扩散在质量上比肩 GAN，是整个扩散浪潮的引爆点。
  - 🏠 官方（TF）：https://github.com/hojonathanho/diffusion
  - ⚙️ 第三方（PyTorch，高星，事实标准）：https://github.com/lucidrains/denoising-diffusion-pytorch

- **Score-based SDE / NCSN (2019–2021, Song 等)** — 用带噪分数匹配 + Langevin 采样，统一为随机微分方程（SDE）框架，把 DDPM 与分数模型纳入同一理论，并给出概率流 ODE（确定性采样）。理论奠基性工作。
  - 🏠 官方（NCSN/score_sde）：https://github.com/yang-song/score_sde ；PyTorch 版：https://github.com/yang-song/score_sde_pytorch

- **DDIM (2020)** — 提出非马尔可夫确定性采样，把采样步数从上千步压到几十步且可复现，是扩散「加速采样」的第一块基石。
  - 🏠 官方：https://github.com/ermongroup/ddim

- **Improved DDPM (2021)** — 学习方差、余弦噪声调度、混合目标等改进，提升似然与样本质量，是工程实践常用基线。
  - 🏠 官方：https://github.com/openai/improved-diffusion

- **ADM / Guided Diffusion (2021)** — "Diffusion Models Beat GANs"。改进 UNet 架构 + 分类器引导（classifier guidance），在 ImageNet 生成上首次全面超越 BigGAN。其 UNet 是后续众多扩散模型（含本仓库 `adm_unet`）的参考架构。
  - 🏠 官方：https://github.com/openai/guided-diffusion

- **Classifier-Free Guidance (2021)** — 用「同一模型联合训练条件/无条件分支、采样时外推」替代独立分类器，成为几乎所有文生图模型默认的可控性技术。
  - ⚙️ 实现（被 GLIDE/SD 等广泛集成）：https://github.com/openai/glide-text2im

- **GLIDE (2021)** — 文本条件扩散 + CFG，是 OpenAI 文生图的早期代表，开源了文本引导扩散的完整管线。
  - 🏠 官方：https://github.com/openai/glide-text2im

- **Latent Diffusion (LDM) / Stable Diffusion (2022)** — 把扩散搬到 VQGAN/VAE 的**低维潜空间**，算力降一到两个数量级，使高分辨率文生图普及化。是开源文生图生态的基石。
  - 🏠 官方（LDM）：https://github.com/CompVis/latent-diffusion ；Stable Diffusion：https://github.com/CompVis/stable-diffusion

- **DALL·E 2 / unCLIP (2022)** — 「CLIP 图像 prior + 扩散解码器」的两阶段文生图，展示强语义对齐与多样性。
  - ⚙️ 第三方（高质量复现）：https://github.com/lucidrains/DALLE2-pytorch

- **Imagen (2022)** — 用大型冻结文本编码器（T5-XXL）+ 级联超分扩散，强调「文本编码器规模 > 扩散模型规模」，文本对齐质量极高。
  - ⚙️ 第三方（高质量复现）：https://github.com/lucidrains/imagen-pytorch

- **EDM (2022, Karras 等) / EDM2 (2024)** — 把扩散的噪声调度、预处理、采样器拆成可独立调参的设计空间，给出工程化最优配置，是当前训练高质量扩散模型的事实标准配方；EDM2 进一步改进网络与 EMA 调度。
  - 🏠 官方：https://github.com/NVlabs/edm ；EDM2：https://github.com/NVlabs/edm2

- **DiT — Diffusion Transformer (2023)** — 用 Transformer（而非 UNet）做扩散骨干，验证良好的 scaling 规律，是 Sora、SD3 等新一代模型的架构基础。
  - 🏠 官方：https://github.com/facebookresearch/DiT

- **Consistency Models (2023)** — 学习「任意噪声点直接映射回数据」的一致性函数，实现**单步/少步**高质量采样，是扩散加速（蒸馏）方向的代表。
  - 🏠 官方：https://github.com/openai/consistency_models

- **Rectified Flow / Flow Matching (2022–2023)** — 用「拉直的概率流 ODE」直接回归速度场，训练更简单、采样更少步，是 SD3、Flux 等最新模型采用的范式（连续归一化流与扩散的统一）。
  - 🏠 官方（Rectified Flow）：https://github.com/gnobitab/RectifiedFlow ；Flow Matching（Meta）：https://github.com/facebookresearch/flow_matching

- **SDXL (2023)** — Stable Diffusion 的大幅升级（双文本编码器、更大 UNet、refiner、尺寸/裁剪条件），开源文生图的高质量基线。
  - 🏠 官方：https://github.com/Stability-AI/generative-models

---

## 6. 任务专项：超分辨率（Super-Resolution）

与本项目（医学图像生成/复原）直接相关。按 GAN 路线与扩散路线两支发展。

- **SRCNN (2014)** — 第一个端到端 CNN 超分，三层卷积学习 LR→HR 映射，开创深度学习超分。
  - ⚙️ 第三方：https://github.com/yjn870/SRCNN-pytorch

- **EDSR (2017)** — 去掉冗余 BatchNorm 的增强残差网络，NTIRE 2017 冠军，是 PSNR 导向超分的经典骨干。
  - 🏠 官方：https://github.com/sanghyun-son/EDSR-PyTorch

- **SRGAN (2017)** — 首次用对抗 + 感知损失追求**视觉真实**而非单纯 PSNR，提出感知超分概念。
  - ⚙️ 第三方（高星）：https://github.com/tensorlayer/srgan

- **ESRGAN (2018) / Real-ESRGAN (2021)** — ESRGAN 用 RRDB + 相对判别器改进细节；Real-ESRGAN 用合成退化建模真实世界复杂降质，是目前实用盲超分/老照片修复的事实标准。
  - 🏠 官方（ESRGAN）：https://github.com/xinntao/ESRGAN ；Real-ESRGAN：https://github.com/xinntao/Real-ESRGAN

- **SwinIR (2021)** — 基于 Swin Transformer 的图像复原通用骨干，在超分/去噪/去压缩伪影上全面领先 CNN。
  - 🏠 官方：https://github.com/JingyunLiang/SwinIR

- **SR3 (2021)** — 用条件扩散做超分（迭代细化），是扩散路线超分的代表，质量超越 GAN 基线。
  - ⚙️ 第三方（高质量复现）：https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement

---

## 7. 任务专项：去噪（Denoising）

- **DnCNN (2017)** — 残差学习（预测噪声）+ BatchNorm 的盲高斯去噪开山之作，单模型可处理多噪声水平，是深度去噪基准。
  - 🏠 官方：https://github.com/cszn/DnCNN （作者维护的统一复原库 KAIR：https://github.com/cszn/KAIR）

- **FFDNet (2018)** — 以噪声水平图为输入，单网络灵活处理空间变化噪声，速度快、可调。
  - 🏠 官方：https://github.com/cszn/FFDNet

- **Noise2Noise (2018)** — 证明**无需干净标签**、仅用成对含噪图即可训练去噪网络，是自监督复原的里程碑。
  - 🏠 官方：https://github.com/NVlabs/noise2noise

- **Restormer (2022)** — 高效 Transformer（通道注意力 + 门控前馈）做高分辨率复原，在去噪/去雨/去模糊上 SOTA。
  - 🏠 官方：https://github.com/swz30/Restormer

- **NAFNet (2022)** — "Nonlinear Activation Free Network"，去掉非线性激活的极简高效复原网络，去噪/去模糊 SOTA 且推理快。
  - 🏠 官方：https://github.com/megvii-research/NAFNet

---

## 8. 任务专项：图像修复（Inpainting）

- **Context Encoders (2016)** — 第一个用对抗 + 重建损失做语义补全的深度方法，开创深度 inpainting。
  - 🏠 官方：https://github.com/pathak22/context-encoder

- **DeepFill v1/v2 (2018/2019)** — 引入上下文注意力（contextual attention）与门控卷积（gated conv）处理不规则掩码，是自由形状修复的经典。
  - 🏠 官方：https://github.com/JiahuiYu/generative_inpainting

- **EdgeConnect (2019)** — 「先补边缘结构、再补纹理」的两阶段法，对结构性缺失修复效果好。
  - 🏠 官方：https://github.com/knazeri/edge-connect

- **LaMa (2021)** — 用快速傅里叶卷积（FFC）获得全图感受野，擅长大面积缺失与重复纹理，是当前实用 inpainting 强基线。
  - 🏠 官方：https://github.com/advimman/lama

- **RePaint (2022)** — 无需为掩码重训：用预训练扩散模型 + 重采样策略做 inpainting，质量高、泛化好。
  - 🏠 官方：https://github.com/andreas128/RePaint

- **Palette (2022)** — 统一的条件扩散框架，单一范式覆盖上色/补全/去噪/JPEG 复原等多种图到图任务。
  - ⚙️ 第三方（高质量复现）：https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models

---

## 9. 常用评测指标

生成质量难以单一标量度量，通常组合使用以下指标：

- **IS（Inception Score）** — 用 Inception 分类置信度衡量清晰度与多样性；对 ImageNet 外数据可靠性差，现多作为辅助。
  - ⚙️ 实现（Improved GAN 内）：https://github.com/openai/improved-gan

- **FID（Fréchet Inception Distance）** — 在 Inception 特征空间比较真实/生成分布的均值协方差距离，是当前最主流的整体质量指标（越低越好）。
  - 🏠 官方（TTUR）：https://github.com/bioinf-jku/TTUR ；⚙️ PyTorch 常用：https://github.com/mseitzer/pytorch-fid ；⚙️ clean-fid（消除重采样/格式偏差）：https://github.com/GaParmar/clean-fid

- **LPIPS（学习感知图像块相似度）** — 用深度特征衡量两图感知距离，广泛用于超分/复原/inpainting 的成对评测（越低越相似）。
  - 🏠 官方：https://github.com/richzhang/PerceptualSimilarity

- **PSNR / SSIM** — 像素级保真度指标，超分/去噪等复原任务的标配（与 LPIPS 互补：高 PSNR 不一定感知好）。
  - ⚙️ 常用实现：scikit-image / torchmetrics 内置。

---

## 10. 综述与延伸阅读

- **What are Diffusion Models?**（Lilian Weng 博客）— 扩散模型最清晰的入门综述之一：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- **Score-based Generative Modeling**（Yang Song 博客）— 分数/SDE 视角统一讲解：https://yang-song.net/blog/2021/score/
- **Hugging Face `diffusers`** — 工业级扩散模型库（DDPM/DDIM/LDM/SD/SDXL/一致性模型等统一 API），落地与复现首选：https://github.com/huggingface/diffusers
- **`lucidrains` 系列** — 大量前沿生成模型的高质量 PyTorch 复现（DDPM、Imagen、DALL·E2、MaskGIT 等）：https://github.com/lucidrains

---

### 对 `gentask` 的选型建议（参考，不在本调研强制范围内）

结合本项目「医学图像、复原/生成类任务（超分/去噪/inpainting）、已有 ADM-UNet / EDM2-UNet 骨干」的现状：

- **稳妥强基线**：回归类任务可先用 **Restormer / NAFNet / SwinIR**（去噪、超分）或 **Real-ESRGAN**（盲超分）作对照。
- **生成/复原统一范式**：用**条件扩散**（SR3 / Palette / RePaint 思路）复用本仓库已有的 UNet 骨干，最契合「一套框架覆盖超分+去噪+inpainting」的目标。
- **训练配方**：扩散训练参考 **EDM/EDM2** 的预处理与噪声调度；评测用 **FID + LPIPS + PSNR/SSIM** 组合。
