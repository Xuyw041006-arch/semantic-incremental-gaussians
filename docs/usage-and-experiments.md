# 从视频到增量语义 Gaussian

输入一个 RGB 视频，自动完成：解码抽帧 → 重复帧过滤 → COLMAP 特征与匹配 → 相机位姿及稀疏点云估计 → 去畸变 → SAM2/CLIP 分割 → 多视角物品与细区域关联 → 有预算的增量 Gaussian 优化 → 地图、轨迹、开销回放及 PLY 导出。

这是原有 S/B/R 语义控制器的独立视频入口，原来的 40 组消融结果保留在另一目录。

## Colab：上传自己的视频

打开 `colab_video_to_gaussians.ipynb`，选择 NVIDIA GPU（本项目使用 L4 验证），顺序运行。源码嵌在笔记本里，上传 MP4/MOV/MKV/AVI 等 PyAV 可解码的视频后执行即可。不需要相机内参、位姿、深度或手工分割标注。不支持的编码会报解码错误，不会伪造帧。

笔记本也提供真实 RGB 录像的 MP4 样例。样例来自 TUM RGB-D 的 RGB 记录，为测试视频入口重新编码；**重建不读取深度或 groundtruth.txt**。这不是合成场景。

## 命令行

在本目录下运行，路径中的空格可以正常使用：

```bash
python install_colab.py
python -m videogs doctor
python -m videogs run --video "/content/room.mp4" --output "/content/room-result"
```

默认每秒约 3 帧，最多 180 帧，抽帧宽 960、训练宽 640，最多 6 个阶段，每阶段 2400 次优化，Gaussian 上限 **500,000**。长视频会增大采样间隔以覆盖整个视频；可用 `--start` / `--duration` 限定片段。实际阶段数会根据早期三角化点数减小，避免一两张图片就强行训练。

默认 `--stability-mode mature` 使用修正后的批次调度：清空跨批次梯度统计；前 200 步积累本批观测；最后 800 步停止增密与裁剪、取消裁剪窗口并均衡回放所有已到达训练视角。子高斯至少经过 200 次全局更新、12 次非零图像梯度观测和两个不同帧后，才可再次增密；父点也有 200 步再次增密冷却。这里的观测是梯度支持代理，不是精确的像素贡献量。短批次会按长度缩短预热和稳定阶段。

`--stability-mode schedule` 只启用批次统计重置、预热和稳定阶段；`legacy` 保留之前的调度以供对照。`--batch-warmup`、`--batch-settle`、`--growth-fraction`、`--min-child-age` 和 `--min-fit-observations` 可调。成熟模式每次增密的候选上限为 `max(2048, 当前高斯数 × growth_fraction)`，默认比例 0.1，仍受剩余容量约束。50 万是上限，不要求用完。修正保留 L1 图像损失和原学习率，不启用此前失败的 DSSIM 巩固。

```bash
python -m videogs run --video room.mp4 --output room-result \
  --fps 3 --max-frames 240 --stages 6 --steps 2400 --cap 500000 \
  --important "monitor,keyboard,mouse,cup,bottle,book,laptop" \
  --background-prune-fraction 0.03
```

`--factors 111` 启用三项语义控制；`101` 启用 S+R；`000` 使用基础控制器。每次对照使用不同输出目录。

## 每批图像的语义与高斯更新

1. 只读取已经到达、且属于训练集的 SAM/CLIP 结果。通过三维重叠和 CLIP 描述关联物品与细区域；保留最多 4 个物品级 512 维 CLIP 描述。默认重点为显示器、键盘、鼠标、杯子、瓶子、书本和笔记本电脑，可用 `--important` 改为英文文字查询。
2. 每个高斯同时保存物品 ID、细区域 ID、置信度、重要程度和边界分数。训练中学习 16 维压缩 CLIP 特征；固定随机投影不使用未来图像的 PCA。每 10 步在已到达教师帧上进行特征监督，并按实际高斯投影和渲染深度更新语义。特征损失只更新语义属性；几何由 RGB 误差和重要性控制器更新，避免不稳定的 CLIP 目标直接拉动位置。
3. 重要物品优先参与选图、裁剪和回放。增密依据屏幕梯度，并对重要物品与其可靠边界加权；大高斯分裂，小高斯复制，可靠边界可优先分裂。增密保留 30% 普通梯度候选机会。所有操作同步调整 Adam 状态，子高斯继承语义。
4. 每批经过至少 300 步证据积累后，只从可靠背景、已有多次可见观测、误差较低的高斯中裁剪贡献代理值最低的尾部；最多裁剪该批背景点的 3%，并保护每个背景空间网格的强代表。未知类别、重点物品、刚生成的高斯不会用于背景预算裁剪。低透明度等常规无效高斯裁剪仍单独进行。某批没有满足条件的背景时，裁剪数可以为 0。
5. 50 万是最终上限，阶段容量随视频到达逐步扩大。`--semantic-densify 0` 关闭任务重要性增密、加权 RGB 与背景预算裁剪，保留语义特征学习，便于匹配预算对照。S/B/R 的开关仍有效。

CLIP 相似度用于识别匹配，不是物品重要程度或校准概率。背景必须有类别和置信度支持；跨视角与部件层级仍是预测。贡献分数由可见屏幕面积、透明度和训练误差估计，并非精确的删除反事实损失。

真实视频演示：

```bash
python -m videogs demo --output videos/tum-desk.mp4 --seconds 24
python -m videogs run --video videos/tum-desk.mp4 --output results/demo
python -m videogs serve --output results/demo --port 8770
```

打开 `http://127.0.0.1:8770/viewer.html`。播放器同步显示输入、逐阶段地图、相机轨迹、Gaussian 数量、优化延迟与显存。新训练保存每阶段的完整高斯；交互窗口使用 WebGL2 各向异性椭圆投影、深度排序和透明度合成，颜色使用 SH DC，下方对照图为完整 SH 的 CUDA 光栅化。默认打开最终阶段并使用真实拍摄视角，可拖动环绕、滚轮靠近、Shift 拖动平移，支持回到拍摄视角、全景适配和物品聚焦。视野外近高斯的投影 Jacobian 有限幅，避免将整个场景遮成灰雾。

旧的 20 万模型已升级显示方式，但没有保存完整模型的旧中间阶段仍会明确显示为中心采样。新版本各阶段都保存完整 `.splat`，不会用重复点冒充更多训练高斯。

`serve` 和笔记本的播放器服务器支持 HTTP Range，Safari / Codex 内置浏览器也可以跳转时间轴。显存数值为优化阶段的 PyTorch 分配峰值，不含 CUDA 驱动上下文和分割模型。

## 当前采用：50 万预算的增量稳定性修正

新模型和三组对照在 `validation/rgb-video-06-stable/`，打开 `RESULTS.html` 查看逐视角图片、指标与比较边界。真实 TUM desk RGB 视频、59 帧（52 训练 / 7 测试）、4 批 × 2,400 步、seed 7、NVIDIA L4，三组都从稀疏点重新训练。

| 同输入 / 同步数设置 | 实际高斯数 | 整体 PSNR | SSIM | 早期测试帧 7 PSNR | 重要物品 PSNR |
|---|---:|---:|---:|---:|---:|
| 原调度重新运行 `legacy` | 498,793 | 19.75 | 0.682 | 16.12 | 21.43 |
| 批次清零 + 预热 + 稳定阶段 `schedule` | 498,648 | 20.54 | 0.697 | 23.21 | 21.39 |
| 再加成熟条件、冷却和渐进增密 `mature` | 498,621 | **22.14** | **0.745** | 22.58 | **23.45** |

采用 `mature`：整体比本次原调度重跑提高 2.39 dB，重要边界 PSNR 从 17.78 到 18.91 dB。最后生成的高斯在保存前经历约 899 次后续全局更新，旧调度约 199 次；这不是每个高斯都获得 899 次有效可见监督。稳定阶段均衡覆盖全部已到达训练帧；最后一批每帧被采样 15–16 次。

优化次数相同，但稳定阶段使用整图而非局部裁剪，修正版总优化像素多约 16%；这不是严格相同 FLOPs 的对照。三组训练耗时分别约 115.0 / 118.2 / 114.6 秒，不含 SfM、教师构建和测试。修正版末批更新 p95 为 25.50 ms，PyTorch 更新峰值 827.03 MiB。

这是单场景、单种子的探索性验证，不代表统计显著的通用提升。帧 7 仅代表一个早期视角，重要物品区域由 SAM/CLIP 预测掩码定义。部分测试视角仍低于 20 dB，细纹理尚有模糊。16 项 CPU 测试与两组实际 CUDA 回归测试通过；完整高斯、Adam 检查点、冻结训练代码、密度事件和采样审计均保存。

```bash
python -m videogs serve --output validation/rgb-video-06-stable --port 8772
```

## 历史：50 万预算与语义重要性验收

新结果在 `validation/rgb-video-04-priority/`，打开其中的 `RESULTS.html` 查看完整对照和逐批日志。使用同一真实 RGB 视频、同一 59 帧划分、4 批 × 2,400 步、seed 7、50 万上限，从稀疏点重新训练。

| 同预算设置 | 实际高斯数 | 整体 PSNR | 重要物品 PSNR | 重要物品轮廓 PSNR |
|---|---:|---:|---:|---:|
| 关闭任务重要性控制 | 498,971 | 17.31 | 17.59 | 16.48 |
| 开启任务重要性控制 | 498,774 | 18.65 | 20.11 | 17.61 |

PSNR 单位为 dB。两组都保留基础 S/B/R 与语义特征学习；对照关闭任务加权采样、增密、RGB 权重及背景预算裁剪，因此这个组合对照不等于各模块独立消融。重要物品与轮廓来自留出图像的 SAM/CLIP 预测掩码，不是人工真值。GPU 计算存在非确定性，单场景和单种子结果不能证明稳定的统计优势。

最终重要物品高斯 216,767 个，其中重要边界高斯 78,576 个；四批背景预算裁剪合计 154 个。末批优化 p95 为 23.69 ms，优化分配峰值为 803.19 MiB。这个显存统计不含分割和相机估计。

旧版 20 万预算、3,200 步的整体 PSNR 为 20.55 dB；此历史版本未全面超过旧版画质。增加点数和步数可能过拟合；应同时检查重要物品、旧区域和整体质量。另试验了 1,500 步的全视角 L1/DSSIM 巩固，语义版本降至 11.20 dB，已弃用并保留失败记录，未纳入生产链路或最终模型。

13 项本地测试及实际 CUDA 检查通过。所有阶段的完整 `.splat`、16 维语义特征、原始指标、分裂/裁剪审计以及带 Adam 状态的完整检查点已保存。本地结果包 SHA256 与 Colab 一致。物品聚焦会先选择观察过该物品的相机；默认场景仍使用最终拍摄视角。

## 旧版 20 万预算的真实视频验收

2026-10-08 在 Colab NVIDIA L4 完成 TUM desk RGB 录像的 MP4 输入验证。输入为 613 帧、约 20.4 秒的真实录像，仅使用 RGB；自动抽取 59 帧，59 帧全部定位，平均重投影误差 0.87 像素。

| 阶段 | 累计帧数 | Gaussian 数 | 留出视角 PSNR |
|---|---:|---:|---:|
| 1 | 15 | 49,577 | 17.35 dB |
| 2 | 30 | 97,789 | 17.63 dB |
| 3 | 44 | 149,389 | 19.04 dB |
| 4 | 59 | 195,534 | 20.55 dB |

最终有非零 Gaussian 支持的物品区域 21 个、细区域候选 35 个；这是预测区域数量，不是人工确认的真实物品数。末阶段优化 p95 为 14.27 ms，PyTorch 优化分配显存峰值 222.04 MiB。六个处理阶段累计约 606 秒，其中包含首次 CUDA 扩展编译 444 秒；不含依赖安装和原录像下载。PSNR 和 SAM 区域一致性均有下文所述评估边界。

完整结果在 `validation/rgb-video-02/`，原始 Colab 回传包为 `validation/rgb-video-02-results.zip`；SHA256 为 `b9d09aaa520baa2332ccbe059a14c5acb6c18833094556df962b434118b781dd`。`code-used.zip` 保存该次训练所用源码。训练后修复的 HTTP Range 服务已经通过本地浏览器验证，模型权重未因此重训。

该次运行参数：

```bash
python -m videogs run --video videos/tum-desk.mp4 --output results/desk-validation \
  --fps 3 --max-frames 72 --extract-width 640 --width 640 \
  --stages 4 --steps 800 --cap 200000
```

本地 8 项检查覆盖真实视频编解码、时间戳、相机数据衔接、训练帧点释放、短片失败诊断、旋转、PLY 属性及 HTTP 分段读取。浏览器验证覆盖阶段跳转、视频完整播放、物品与细区域显示。单个场景验收不能代表所有手机视频均能重建成功。

## Apple Silicon / 无 GPU

本地可以完成视频解码和 CPU COLMAP；完整 Gaussian 优化与 SAM2 推理使用 NVIDIA CUDA：

```bash
python -m pip install -e .
python -m videogs prepare --video room.mp4 --output room-prepared
```

`prepare` 完成后状态是 `ready_for_gpu`，不会声称已训练。要从准备阶段继续，使用同样的视频、参数和路径运行 `run`。跨电脑移动数据应使用新的输出目录，或在 Colab 从视频重新执行。

## 输出

| 文件 | 含义 |
|---|---|
| `viewer.html`、`input-preview.mp4` | 可播放的建图回放；预览是去畸变后的抽帧视频 |
| `gaussians.ply` | 标准 3DGS 属性（球谐、对数尺度、logit 透明度、四元数）及物品/细区域 ID |
| `training/map.npz`、`training/checkpoint.pt` | Gaussian 参数、16 维语义特征和最终检查点（含 Adam 状态） |
| `training/stage-*.splat` | 每阶段完整高斯：位置、尺度、透明度、四元数、DC 颜色及两级 ID |
| `training/semantic-state.npz`、`training/density-events.json` | 每高斯的重要性/边界/背景分数，以及分裂、复制、背景裁剪审计 |
| `prefixes/semantic-frame-*.npz` | 已到达训练教师的两级区域、置信度和边界 |
| `gaussian-semantic-labels.npz`、`objects.json` | 每个 Gaussian 的层级区域 ID、名称候选和父子关系 |
| `camera-trajectory.json` | 相机轨迹与内参；单目尺度不能直接解释为米 |
| `sfm/sfm-quality.json` | 定位率、丢失帧、重投影误差和重试信息 |
| `summary.json`、`pipeline.json` | 分阶段耗时、配置、输入/源码校验值及运行状态 |
| `logs/` | 分割、语义关联、训练及失败日志 |

输出目录绑定视频内容、参数及源码摘要。原命令重跑会跳过已完成阶段；**训练中断后会从该训练阶段重新开始，不是恢复 Adam 状态**。改变视频、预算或代码时必须使用新目录，防止混用旧缓存。Colab 临时磁盘可能被回收，应下载结果包；仅关闭网页后重新连接仍存在的运行时，不保证临时文件永久保存。

## 数据质量与边界

- 相机要有平移、足够的相邻视角重叠和可辨认纹理。纯旋转、极度模糊、大片白墙或大量运动物体可能导致 SfM 失败。
- 自动使用一个共享镜头模型。超广角/鱼眼可以选择 `--camera-model OPENCV_FISHEYE`；镜头切换、变焦或视频剪辑应拆开处理。
- 默认要求至少 75% 抽帧成功定位、12 张注册图片和 100 个稀疏点。短视频失败时最多尝试一次全匹配；失败结果会保留诊断，不生成“成功”的地图。
- 相机位姿由完整视频离线估计，再按时间前缀更新 Gaussian。这条链路**不是严格在线 SLAM**，也没有从视频直接恢复物理米制尺度。
- 留出的图片不参与 Gaussian 拟合，但参与 SfM。因此 PSNR 表示“在同一离线估计几何下的留出视角重建”，不能称为完全无泄漏的相机估计评估。
- SAM 区域、CLIP 类别名以及跨视角 ID 是预测。细层是较小区域候选，不保证是语义正确的椅腿、把手等部件。
- 优化 p95 / 显存单独统计，分割、SfM 与导出单列耗时，回放速度不能当作端到端实时速度。

## 主要来源

- [COLMAP CLI / SfM 与去畸变](https://colmap.github.io/cli.html)
- [官方 COLMAP Python 接口](https://github.com/colmap/colmap/tree/main/python)
- [SAM2 官方实现](https://github.com/facebookresearch/sam2)，固定提交 `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- [gsplat](https://github.com/nerfstudio-project/gsplat)，固定 `1.5.3`
- [TUM RGB-D 数据集](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

模型和原始数据按各自许可使用；本包保留原有 gsplat 许可文件。
