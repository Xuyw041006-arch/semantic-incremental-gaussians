# 语义驱动的增量 Gaussian：S / B / R 消融

本目录实现三个可独立关闭的控制器：语义选图及裁剪 S、层级区域预算 B、覆盖回放 R。几何仍通过同一 RGB/深度损失优化；语义决定有限计算和高斯容量分配到哪里。

`EXPERIMENT_REPORT.md` 在 32 组实际运行及检查结束后生成，包含完整结果、负面结果和边界。`analysis/` 保存表格与图；`results/` 保存原始日志和指标。

## 运行位置

当前实验在[原 Colab](https://colab.research.google.com/drive/1pJRfsBYrjzoTGzjTftAHG2kXYamMAEcx)最后新增的 Semantic innovations/ablations 单元格。运行根目录 `/content/semantic-incremental`，与旧实验隔离。

使用新的 NVIDIA Colab 运行时复现：上传本目录源码，或使用带嵌入源码的 `colab_semantic_ablations.ipynb`。

```bash
python bootstrap_colab.py
python prepare_data.py
python run_ablation.py
python analyze_results.py --checkpoints
python benchmark_teacher.py
```

精确复用本次输入时，将结果归档中的 `data/*/teachers` 和 `data/*/prefixes` 放回本目录；重新生成 2D 教师会引入另一次数据准备差异。厨房的 41 个训练缓存来自此前宽 779 的教师提取，统一缩放到本次宽 640。其他教师视图本次实际运行生成，全部组合共享固定缓存。代码可从头生成缓存，但这不应被称为逐位重现本次输入。

TUM 导入器先取每 5 个 RGB 帧，关联 RGB/深度/真值位姿，最多 120 帧，实际 119 帧。厨房真实照片 279 张，按文件名顺序分六批。已关联位姿的通用 RGB-D manifest 入口参见 `hglab/readers.py`，必须使用 OpenCV c2w 米制位姿。

单组运行例子（`111` 中三位依次是 S、B、R）：

```bash
python -m hglab.incremental --data data/tum640/manifest.json --prefixes data/tum-desk/prefixes --masks data/tum-desk/teachers --output results/custom/111/seed-7 --factors 111 --seed 7 --cap 300000 --stages 6 --steps 600 --width 640
```

原协议三种种子 7/17/27；用户授权减少种子后，在查看汇总前修订为两种种子 7/17，全部八个开关组合，每前缀 600 步，共 3,600 步，宽 640，上限逐步升至 30 万。采样审计、增密/裁剪事件和每视角指标都随结果保存。重启矩阵会跳过有完整 `run.json` 的组，未完成组从头重跑；没有实现组内优化器检查点恢复。不要复用有完成标记的目录进行不同配置训练。

## 实现入口

- `online_semantics.py`：仅使用已到达训练视图生成前缀候选区域图，持久节点、双尺度标签及有界 CLIP 原型。
- `policies.py`：S 的选图权重/区域裁剪，B 的稀缺区域分数，R 的加权覆盖选择及探索保留。
- `incremental.py`：真实 gsplat CUDA 优化，B 的高斯准入、克隆/分裂/裁剪，区域身份继承。
- `evaluation.py`：留出图像 PSNR/SSIM、SAM 区域 Hungarian 匹配和固定早期相机遗忘指标。
- `analyze_results.py`：配置矩阵审计、配对去模块效应、平均主效应及二阶交互；种子为统计重复单位。
- `finish_experiment.py`：等待矩阵完成，自动测试、审计、单独教师基准、报告和归档。

`base_train.py` 为保持与实际运行源码一致而保留旧训练器，但本版仅复用其渲染、初始化、追加和导出函数；请使用上面的 `hglab.incremental` 入口。

## 解释边界

这是两个真实场景的已知位姿、六批增量建图。TUM 位姿为真值；厨房 COLMAP 使用离线全局信息，不能声称严格因果 RGB SLAM。分割参考为 SAM 预测区域，细区域只是候选部件，CLIP 名称和包含关系可能出错。本版没有在线训练 SAGA 式亲和场，不是 SAGA/LaGa 完整复现。

控制器作用已接入训练，但是否有效由消融决定。固定更新次数不等于固定耗时，更不等于端到端实时；SAM/CLIP、前缀图和评估成本独立记录。高斯/图像缓存/回放池有界，当前全量 CPU 锚点和历史元数据并非无限流的恒定内存实现。

gsplat 策略相关许可保存在 `GSPLAT_LICENSE.txt`。数据与外部模型遵循各自官方来源许可。
