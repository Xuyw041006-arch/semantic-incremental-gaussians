# Semantic Incremental 3D Gaussian Reconstruction

[简体中文](README.md)

**Turn indoor RGB video into a semantic 3D scene that grows through budgeted incremental updates.**

`3D Gaussian Splatting` · `SAM2` · `CLIP` · `COLMAP` · `PyTorch / CUDA` · `WebGL2`

![Real RGB video, Gaussian reconstruction, object-level regions and fine-region candidates](docs/assets/project-overview.jpg)

*Final-stage training view of a real TUM desk scene. Semantic colors encode predicted region IDs; fine regions are hierarchy candidates.*

## Overview

This project implements a complete **video-to-3D pipeline**: decoding, camera estimation, semantic segmentation, incremental Gaussian optimization, export and visualization. It accepts indoor RGB video without user-supplied depth, camera poses or segmentation annotations.

Semantics guide reconstruction decisions: **which views to train on, where to densify, how to allocate limited capacity, and which past observations to replay**. The map stores both appearance and semantics for object inspection and hierarchical region visualization.

## Features

| Capability | Implementation and value |
|---|---|
| Video input | Accepts decodable MP4, MOV, MKV and AVI files, with frame sampling, near-duplicate filtering and automatic camera/sparse-structure estimation. |
| Two-level semantic association | Combines SAM2 regions, CLIP descriptors and 3D overlap to maintain object-level and fine-region IDs across views. |
| Task-defined object importance | English object queries, such as monitors, keyboards and cups, define reconstruction priorities. |
| Budgeted incremental updates | Extends the existing map batch by batch, progressively releasing capacity up to a default **500,000-Gaussian** cap. |
| Stable densification | Gates repeated growth by age, multi-view gradient support and cooldown, then refines with fixed topology at each batch tail. |
| Export and playback | Exports PLY, the full model, semantics, trajectories and stage maps; playback synchronizes video, map growth and compute metrics. |

## Design contributions

### 1. Semantics direct the compute budget

Three controls operate throughout reconstruction, rather than only labeling a finished map.

| Module | Mechanism | Purpose |
|---|---|---|
| **S: Semantic sampling** | Mixes RGB difficulty, semantic novelty and task importance when selecting views and crops, while retaining uniform exploration. | Spend optimization updates on informative observations. |
| **B: Budget allocation** | Prioritizes important objects and reliable boundaries for gradient-based growth; conservatively prunes supported, low-contribution background while retaining spatial representatives. | Balance task detail and scene coverage within limited capacity. |
| **R: Semantic replay** | Selects past views using region coverage, importance and pose diversity, with temporal coverage and balanced sampling during settling. | Reduce forgetting as new areas arrive. |

### 2. Learn semantic attributes during reconstruction

Each Gaussian carries a **16-dimensional compressed CLIP feature**, two-level IDs, confidence, importance and boundary attributes. Object nodes retain a bounded bank of multi-view CLIP descriptors. Feature supervision updates semantic attributes; RGB fitting and the semantic controller guide geometry and appearance. New Gaussians inherit semantics, and topology operations maintain their attributes.

### 3. Give new Gaussians time to fit

Each batch defaults to **2,400 updates: 200 warmup + 1,400 refinement/topology updates + 800 settling updates**. New Gaussians must satisfy maturity conditions, and parents have a growth cooldown. The batch tail freezes topology while continuing to fit arrived views. Capacity is a ceiling, and growth remains evidence-driven.

These are implemented project designs and engineering contributions built on existing open-source methods.

## How incremental reconstruction works

![Video-to-semantic-3D processing pipeline](docs/assets/pipeline.png)

1. **Initialize:** estimate cameras and sparse structure using COLMAP, releasing initialization points according to observation order.
2. **Add a batch:** reuse Gaussian parameters and update controls from arrived training views and semantic teachers.
3. **Sample and fit:** mix new observations with replay, render the current map, compare against RGB targets and backpropagate.
4. **Grow and prune:** split or clone using gradients, importance, boundary reliability and maturity; prune background only when evidence supports it.
5. **Settle and save:** freeze topology, refine arrived views with balanced sampling, and save the stage map, quality and costs.

An optimization update samples one view or image crop; **it is not an epoch over the entire dataset**. Each new batch continues optimizing the existing map without restarting the whole reconstruction.

## Real-scene validation

Validated on a **real TUM desk RGB recording**, without depth or ground-truth trajectories as video-pipeline inputs. **NVIDIA L4**, seed **7**, **59** registered frames, **52** Gaussian-fitting views and **7** held-out views, **4 batches × 2,400 updates**.

| Metric | Measured result |
|---|---:|
| Final Gaussian count / cap | **498,621 / 500,000** |
| Mean held-out PSNR | **22.14 dB** |
| Mean held-out SSIM | **0.745** |
| Important-object region PSNR | **23.45 dB** |
| Final-batch optimization update latency p95 | **25.50 ms** |
| Optimization peak PyTorch allocation | **827.03 MiB** |
| Total optimization time across four batches | **114.58 s** |
| Camera registration | **59 / 59** |

| Arrived frames | 15 | 30 | 44 | 59 |
|---|---:|---:|---:|---:|
| Gaussian count | 27,766 | 105,908 | 332,632 | 498,621 |

![Held-out frame 7 and final-model reconstruction](docs/assets/heldout-comparison.jpg)

*Left: held-out frame 7. Right: final-model rendering, with 22.58 dB PSNR for this view.*

Recorded validation includes **16 CPU tests and 2 CUDA regression groups passed**, covering stage counts, finite parameters, semantic IDs, Adam state and fixed topology during settling. See the [run summary](docs/evidence/run-summary.json), [checkpoint validation](docs/evidence/checkpoint-validation.json) and [incremental checks](docs/evidence/incremental-validation.json).

The repository also preserves code, metrics and analyses for [S/B/R factorial ablations and separate-recording verification](research/semantic-incremental-v2/README.md). Those research experiments use known poses and have a different setup from the RGB-video validation above; their conclusions and limitations remain in their own records.

## Quick start

### Colab

Open [colab_video_to_gaussians.ipynb](colab_video_to_gaussians.ipynb), select an NVIDIA GPU, run the cells in order and upload a video. The notebook embeds source code and offers a real-recording example.

### Command line

Run from the repository root in a CUDA/PyTorch environment:

```bash
python -m pip install -e .
python install_colab.py
python -m videogs doctor

python -m videogs run --video room.mp4 --output results/room \
  --stages 4 --steps 2400 --cap 500000 \
  --important "monitor,keyboard,mouse,cup,bottle,book,laptop"
```

Outputs include the full model, PLY map, hierarchical semantics, camera trajectory, stage maps and quality/cost logs. Optional local playback:

```bash
python -m videogs serve --output results/room --port 8770
```

Open `http://127.0.0.1:8770/viewer.html` in a browser.

Without CUDA, `python -m videogs prepare --video room.mp4 --output results/prepared` supports video and SfM preparation. Full optimization and SAM2 inference require an NVIDIA GPU.

## Repository and experiment artifacts

```text
videogs/                      # Video, SfM, pipeline, export and playback
hglab/                        # Incremental training, semantic learning and S/B/R
tests/                        # Video, semantic and refinement checks
docs/                         # Static figures, usage notes and validation evidence
experiments/                  # Recorded video-experiment metrics and checks
research/                     # S/B/R ablations, hierarchy and early research code
showcase/                     # Optional portfolio-page implementation
colab_video_to_gaussians.ipynb # Executable Colab entry point
README.md                     # Chinese introduction
```

Full models, checkpoints, stage maps, experiment archives and showcase media are distributed through [Releases](https://github.com/Xuyw041006-arch/semantic-incremental-gaussians/releases). See [release-assets.json](docs/evidence/release-assets.json) and [SHA256SUMS.txt](docs/evidence/SHA256SUMS.txt) for the inventory and checksums. The Git repository retains code, static figures and metrics while keeping large weights out of Git history.

See [usage and experiment notes](docs/usage-and-experiments.md) for parameters, output formats and interruption behavior. Historical research modules run independently from their own directories; follow their instructions to avoid mixing identically named Python packages with the main entry point.

## Evaluation scope

- **Offline camera estimation precedes prefix-based Gaussian fitting.** Held-out views participate in SfM but not Gaussian fitting; strict online SLAM is not implemented.
- **Latency and memory measure optimization.** They exclude SfM, semantic teachers, installation, first compilation and export, and do not establish end-to-end real-time reconstruction.
- **Semantic labels and hierarchy are predictions.** Region metrics use SAM2/CLIP masks; fine regions are not guaranteed to be human-defined object parts.
- **The video results above cover one scene and seed.** Generalization and consistent gains require broader evaluation. Monocular scale is relative.

## Acknowledgements

Built with [COLMAP](https://github.com/colmap/colmap), [gsplat](https://github.com/nerfstudio-project/gsplat), [SAM2](https://github.com/facebookresearch/sam2) and [OpenCLIP](https://github.com/mlfoundations/open_clip). Real RGB examples originate from the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

Third-party code, models and data remain subject to their respective licenses; the [gsplat license](GSPLAT_LICENSE.txt) is retained.
