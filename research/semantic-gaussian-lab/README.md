# 增量式语义 Gaussian 场景重建 · v0.1

可运行的研究起点：RGB-D 帧持续进入，地图逐帧融合，显示 Gaussian 地图、相机轨迹、语义类别、跨帧物品 ID 和计算开销。完整研究方案见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

**真实实验已完成：**Colab Tesla T4 上运行了 TUM desk 的 119 帧真实 RGB-D、Mask2Former 语义/物品分割和 CUDA Gaussian 优化。99 帧建图、20 帧留出；24k 组 PSNR 14.21 dB，更新 p50/p95 33.7/182.1 ms。完整结果见 [VALIDATION.md](VALIDATION.md)。

**当前边界：**各向同性 Gaussian，使用输入 groundtruth 位姿，尚非完整 SLAM。分割预计算，不能宣称端到端实时。旧视角仍退化约 2.98 dB，物品 ID 碎片化，细节不足。默认合成演示的深度/位姿/分割为真值；查看真实实验请使用下方回放。

## 本地运行

```bash
cd semantic-gaussian-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m sglab serve
```

在浏览器打开 <http://127.0.0.1:8765>。点击播放，可切换外观 / 语义 / 物品实例、二维分割画面，拖动旋转地图，点击类别或物品筛选。调整数量上限或更新时间后，点击“重建”应用。服务仅监听本机。

当前电脑也可以使用已存在的 Codex Python 环境，无需安装依赖：

```bash
/Users/xyw/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m sglab serve
```

## 已实现

- RGB-D 反投影、OpenCV 相机坐标约定与输入校验。
- 体素哈希去重、增量均值融合；每个体素每帧至多增加一次观测次数。
- 固定 Gaussian 数量上限、低支持陈旧点清理、稳定区域冻结。
- 时间软预算：批次截止检查 + 下一帧采样量调整；记录实际超时，不能保证硬实时。
- 每 Gaussian 语义证据和置信度；低置信度输入保持未知。
- 同类别三维体素重叠 / 中心距离关联，帧内实例 ID 转为地图实例 ID；实例元数据有数量和体素上限。
- Mask2Former COCO panoptic 适配：stuff 类别、thing 实例、unknown、预测置信度；人物类别从静态地图过滤。
- TUM RGB-D 导入：时间戳关联、米制深度、真值相机位姿、对齐缩放。
- 可选 CUDA 局部优化：RGB + 深度 + 语义损失、早期关键帧回放、冻结非活动参数。
- NPZ 原始地图、带标准 3DGS 属性和语义/实例 ID 的 PLY、JSON 性能日志。

## 已执行的 Colab 与真实地图回放

[已执行的云端 notebook](https://colab.research.google.com/drive/1pJRfsBYrjzoTGzjTftAHG2kXYamMAEcx)。独立运行可上传 [colab_real_experiment.ipynb](colab_real_experiment.ipynb)，项目代码已内嵌，无需第二次上传。选择 GPU，依次执行检查、真实数据下载、分割、四组独立计时、单独记录回放、完整渲染评估与结果下载。首次 CUDA 编译需要数分钟。

本机已保存实际优化地图和输入，双击 `run_real_replay.command`，或运行：

```bash
python -m sglab.replay --manifest ../real-experiment/data/tum-desk/panoptic.json --result ../real-experiment/results/cuda-replay-24000 --port 8766
```

打开 <http://127.0.0.1:8766/>。回放读取实际优化后快照，不重新建图；可切换 RGB/语义/物品实例，同步查看真实输入、轨迹和记录运行的实际开销。基准性能以 `comparison.json` 为准。

其他 RGB-D 数据入口：

```bash
python -m sglab import-tum --dataset /path/to/tum --output data/tum --frames 100
pip install -e '.[segmentation]'
python -m sglab segment --manifest data/tum/manifest.json --output data/tum/panoptic.json --device cuda
pip install 'gsplat>=1.5,<2'
python -m sglab.real_experiment --manifest data/tum/panoptic.json --output results/tum
# 回放记录另跑一次，避免干扰性能比较
python -m sglab.real_experiment --manifest data/tum/panoptic.json --output results/tum-replay --record-replay
```

旧 [colab.ipynb](colab.ipynb) 为小规模调试入口，真实对照以新 notebook 为准。输入仍需深度与位姿，任意 MP4 暂不能直接完成完整 SLAM。

## 通用数据格式

所有 RGB、深度和分割必须像素对齐。支持 PNG 和无 pickle 的 NPY；浮点深度以米计，整型深度通过 `depth_scale` 转换。位姿为 **camera-to-world**，相机 x 向右、y 向下、z 向前，米制、右手坐标系；世界坐标可选，但本地查看器默认 Y 向上。暂不支持畸变相机或非零 skew。

```json
{
  "name": "my-room",
  "pose_convention": "opencv_c2w_meters",
  "depth_scale": 0.001,
  "K": [[525,0,319.5],[0,525,239.5],[0,0,1]],
  "labels": [
    {"name":"unknown","zh":"未知","color":[113,123,137],"thing":false},
    {"name":"chair","zh":"椅子","color":[102,195,167],"thing":true}
  ],
  "frames": [{
    "timestamp": 0.0,
    "rgb": "rgb/000.png",
    "depth": "depth/000.png",
    "c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    "semantic": "semantic/000.npy",
    "instance": "instance/000.npy",
    "confidence": "confidence/000.npy",
    "segmentation_source": "provided"
  }]
}
```

`labels` 的数组下标就是语义 ID，0 保留为 unknown。`instance=0` 表示 stuff/未知，正数是**帧内**实例 ID；不要求跨帧相同。`confidence` 是 [0,1] 浮点 NPY，缺省时已知标签置信度为 1。没有分割的帧仍可建几何地图，语义保持未知。位姿必须提供，不会自动从 MP4 估计。

## 验证与日志解释

```bash
python -m unittest discover -s tests -v
python -m sglab run --frames 100 --output results/reference --no-adaptive-budget
python -m sglab.experiments --output results/ablations
```

`map_used_mib` 为已用 Gaussian 数组字节，`map_capacity_mib` 为预分配数组字节；二者都不含 Python 哈希索引、分割模型和帧缓存。`process_peak_rss_mib` 是整个进程历来峰值 RSS，不是当前占用。CPU 后端 `gpu_allocated_mib=null`；CUDA 记录实测 allocated / peak。CUDA 优化额外耗时放在 `refine_ms`，总建图耗时为 `mapping_total_ms`。

旧 `python -m sglab run` 入口的旧视角 probe 只在深度一致的 Gaussian 中心投影上比较颜色、深度，并单独报告覆盖率，**不是完整图像渲染 PSNR**。冻结测试能证明冻结参数不被改写，不能证明真实场景整体质量没有下降。正式评估方案见研究计划。

## 后续实现

RGB-only 深度/位姿适配、相机跟踪与回环、各向异性 Gaussian、在线异步分割、动态物体检测与剔除、子地图磁盘换入换出、可靠的长期实例合并、三维 PQ/mIoU 评估尚未完成。当前历史日志与相机轨迹随序列增长，长时间运行需流式写盘；地图容量上限不等于整个系统 RAM 或 VRAM 的硬上限。

## 主要参考

- [SplaTAM：RGB-D Gaussian SLAM](https://github.com/spla-tam/SplaTAM)
- [RTG-SLAM：紧凑表示与 stable/unstable 在线更新](https://github.com/MisEty/RTG-SLAM)
- [SGS-SLAM：语义 Gaussian SLAM](https://github.com/ShuhongLL/SGS-SLAM)
- [Mask2Former 官方 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/mask2former)
- [gsplat rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html)
- [TUM RGB-D 格式与标定约定](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)
