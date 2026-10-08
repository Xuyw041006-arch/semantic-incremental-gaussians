# 真实室内的增量 Gaussian 与多粒度语义

这版使用 **Mip-NeRF 360 kitchen 的 279 张真实采集照片**，替换小范围 TUM desk 演示。支持几十万点的各向异性椭球、旋转、SH2 颜色、自适应增密和硬容量上限。已在 Colab T4 实际训练，具体结果见 `EXPERIMENT_REPORT.md`。

默认演示使用实测质量较稳的 30 万上限地图；结果包同时提供 60 万上限地图和容量 / 回放消融。增加上限可以继续扩展，例如 `--cap 1000000`，但 **100 万配置尚未实测**。

## 分割实现

- SAM 2.1 Tiny 提取包含关系和不同尺度的候选掩码，不强制压成单层 panoptic 标签。
- 根据 Gaussian 渲染深度估计掩码的三维尺度，并按场景尺度分布校准粗、中、细三档。
- 每个 Gaussian 学习 16 维亲和特征；尺度门控和掩码对比训练参考 SAGA。特征导出为 FP16，浏览器支持点击和连续尺度查询，可调相似度阈值。当前点选可能碎片化，不保证严格嵌套。
- 结合可见性、三维重叠、亲和特征和 CLIP 描述关联跨视角区域。物体节点保留多个视角的语言原型，用紧致度及全局一致性加权，参考 LaGa。
- 区域节点建立有证据支持的父子关系；无法确定的父级保留为未知。细粒度区域是候选部件，名称是 CLIP 预测。

这是借鉴论文机制的实现，**不是 SAGA / LaGa 官方代码的完整复现**。

## 运行

已运行的 Colab：[实验与实测输出](https://colab.research.google.com/drive/1pJRfsBYrjzoTGzjTftAHG2kXYamMAEcx)。

从头复现时，在 Colab 上传 `colab_hierarchical_kitchen.ipynb`，选择 NVIDIA GPU，顺序执行。源码已嵌入，无需再上传代码。首次 gsplat CUDA 编译可能耗时数分钟。

```bash
# 相同协议的容量和回放对照
python -m hglab.train --data /content/mip360/kitchen --output results/kitchen-300k-l1 --cap 300000 --width 800 --stages 6 --steps 1800
python -m hglab.train --data /content/mip360/kitchen --output results/kitchen-600k-l1 --cap 600000 --width 800 --stages 6 --steps 1800
python -m hglab.train --data /content/mip360/kitchen --output results/kitchen-600k-no-replay --cap 600000 --replay 0 --width 800 --stages 6 --steps 1800

python -m hglab.extract_masks --data /content/mip360/kitchen --output semantics
python -m hglab.semantic_field --checkpoint results/kitchen-300k-l1/checkpoint.pt --masks semantics --output semantic-field-300k --steps 1200

# 任意文本查询需要 Colab 的 CLIP 编码器
python -m hglab.query --field semantic-field-300k --text 'a wooden chair' --scale 0.2 --output chair-mask.npy
```

将结果包解压为同级的 `hierarchical-kitchen-results` 文件夹。Mac 可双击 `run_replay.command`；其他环境安装 NumPy / Pillow 后，在代码目录运行：

```bash
python -m hglab.viewer --data ../hierarchical-kitchen-results --port 8767
```

打开 `http://127.0.0.1:8767/`，播放照片序列、观察六批地图扩展和轨迹；在完整地图阶段切换语义粒度、点选查询和预设文本查询。页面顶部为完整 Gaussian 地图的真实 CUDA 渲染，交互视图显示 6 万点 LOD。

## 实验边界

该场景是真实室内的物体中心环绕采集，主要观察桌面、玩具工程车及周围环境；还不能代表长距离或多房间探索。相机位姿及三角化坐标来自离线 COLMAP。只有至少两个已到达训练视角观察到的点才会进入地图，RGB 训练和增密目标仅使用当前批次及有界旧关键帧。这是 **已知位姿下的增量建图实验**，还不是完全在线 SLAM；离线 SfM 初始化也不应视为严格因果输入。

照片按文件名分六批，每第八张排除训练；没有原视频时间戳，不声称端到端实时。每批 1,800 步，共 10,800 步；优化器步骤耗时不等于一帧处理耗时。最多缓存 32 张 RGB，旧关键帧 reservoir 上限 48。

语义场目前在几何完成后训练，历史批次只显示几何。要完成联合在线更新，还需让语义特征和节点 ID 随 Gaussian 增密、裁剪和重定位继承，并验证语义遗忘。当前数据无人工语义 / 部件标注，因此不报告 mIoU、PQ 或真实物体数量。

PSNR / SSIM 对完整保留图像计算；固定第一批的保留视角观察旧区域退化。语义验证是未参与亲和训练的 SAM 关键帧上的像素对一致性，不是人工语义准确率。当前结果是单场景、单随机种子的探索实验。

## 来源

- [Mip-NeRF 360 官方数据](https://jonbarron.info/mipnerf360/)
- [SAGA 官方实现](https://github.com/Jumpat/SegAnyGAussians)
- [LaGa 官方实现](https://github.com/SJTU-DeepVisionLab/LaGa)
- [Hi-LSplat：层级语言 Gaussian](https://arxiv.org/abs/2506.06822)
- [SAM 2 官方实现](https://github.com/facebookresearch/sam2)
- [gsplat](https://github.com/nerfstudio-project/gsplat)，训练策略参考其官方示例；许可保留在 `GSPLAT_LICENSE.txt`。

后续应扩展到多个真实房间和有人工语义 / 部件标注的数据，比较 SAGA / LaGa 原版及联合在线更新，不能凭这一场景宣称达到先进方法水平。
