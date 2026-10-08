"""Generate an evidence-based report; retains negative findings."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def main():
    summary=json.loads((ROOT/'analysis/summary.json').read_text());contrast=json.loads((ROOT/'analysis/contrasts.json').read_text())
    audit=json.loads((ROOT/'analysis/audit.json').read_text());teacher=json.loads((ROOT/'teacher-benchmark.json').read_text())
    prefix=json.loads((ROOT/'analysis/prefix-audit.json').read_text());provenance=json.loads((ROOT/'experiment-provenance.json').read_text())
    plan=json.loads((ROOT/'experiment-plan.json').read_text());n=audit['runs'];ns=len(plan['seeds'])
    assert audit['passed'] and n==len(plan['datasets'])*ns*8
    lines=['# 三项语义控制机制：真实数据消融结果','',
        f'{n} 次独立 CUDA 训练：2 个真实场景 × 8 种 S/B/R 开关组合 × {ns} 个配对随机种子。下面均为均值 ± 样本标准差；只有 {ns} 个种子，不作显著性或 SOTA 声明。用户允许减少种子后，在查看汇总前保留前两个完整种子；原协议及修订记录随结果保存。','',
        'S：语义新颖性和训练 RGB 残差引导选图，并把等尺寸裁剪更多放在已知语义区域。',
        'B：高可信物体及细区域的覆盖稀缺度，影响初始点准入、克隆/分裂排序和容量置换；保留区域代表点。',
        'R：旧帧池按物体/细区域加权覆盖选取，同时保留均匀探索位置与位姿多样性。','',
        '这些是待验证的候选方法，当前实现不是 SAGA/LaGa 官方完整复现。','',
        '## 公平对照协议','',
        '- T4；宽度 640；SH2；6 个数据前缀，每前缀 600 步，共 3,600 步。',
        '- 所有组合共享相同 SAM2.1 Tiny/CLIP 教师缓存与前缀图；000 也有语义输出，但三个控制器均关闭。',
        '- 高斯硬上限逐阶段从 5 万增加到 30 万；真实点数另报。回放概率 .3、旧帧池 48、CPU 图像缓存 32。',
        '- 同种子的每阶段裁剪/整图和回放 Bernoulli 决策相同，裁剪 384×256；审计实际优化像素总数相等。',
        '- 所有组合均有 2% 的定期容量置换机会；仅 B 开关改变置换评分和保护规则。',
        '- 语义教师只从已到达训练图像读入；留出的 RGB/掩码不进入优化目标。厨房的离线 SfM 初值仍可能包含这些视角的信息。高斯分裂/克隆/删除保留锚点身份。',
        '- TUM 每第 6 帧（余数 3）留出；厨房每第 8 帧（余数 0）留出。SAM 参考视图也不进入训练。','',
        '## 主要结果','',
        '细区域分数是 Hungarian 匹配后的 **SAM 区域一致性**，包括未知和未匹配区域的零分；不是人工语义类别或部件真值 mIoU。旧 PSNR 变化是在相同第一阶段留出相机上计算末阶段减首阶段。必须同时看旧视角末阶段绝对分数和阶段曲线：较低的起点也会产生较大的正变化。','']
    names={'000':'基线','001':'R','010':'B','011':'B+R','100':'S','101':'S+R','110':'S+B','111':'S+B+R'}
    def fmt(v,d=3):return f'{v["mean"]:.{d}f} ± {v["sd"]:.{d}f}'
    for ds,variants in summary.items():
        lines+=['### '+ds,'','| 组合 | PSNR dB | SSIM | 物体 SAM | 细区域 SAM | 小细区域 SAM | 旧视角末期 PSNR dB | 旧 PSNR 变化 dB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for f,v in variants.items():lines.append('| '+names[f]+' ('+f+') | '+' | '.join(fmt(v[k]) for k in ['psnr','ssim','object_sam_miou','fine_sam_miou','small_fine_sam_miou','old_psnr','old_psnr_change'])+' |')
        lines+=['','| 组合 | 点数 | 建图总秒 | 最末阶段更新 p95 ms | PyTorch 优化循环峰值 MiB | CPU 峰值 MiB | 全流程进程秒 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for f,v in variants.items():lines.append('| '+names[f]+' | '+' | '.join(fmt(v[k],1) for k in ['gaussians','mapping_seconds','update_p95_ms','gpu_update_peak_mib','cpu_rss_peak_mib','process_seconds'])+' |')
        lines+=['',f'![{ds} 完整因子消融](analysis/{ds}-factorial.png)','',f'![{ds} 旧区域与容量曲线](analysis/{ds}-stages.png)','',
            f'![{ds} 固定早期留出视角实测对比](analysis/{ds}-visual-early.png)','',
            f'![{ds} 后期留出视角实测对比](analysis/{ds}-visual-late.png)','']
        lines+=['**配对消融差值**（完整模型减去对应模块关闭的模型；PSNR/区域分数正值更好，耗时正值更慢）：','',
            '| 对照 | PSNR Δ dB | 细区域 SAM Δ | 旧视角末期 PSNR Δ dB | 更新 p95 Δ ms |','|---|---:|---:|---:|---:|']
        for tag,label in [('111_minus_000','完整 − 基线'),('111_minus_011_S','S 的作用（B/R 开）'),('111_minus_101_B','B 的作用（S/R 开）'),('111_minus_110_R','R 的作用（S/B 开）')]:
            entries={r['metric']:r for r in contrast if r['dataset']==ds and r['contrast']==tag}
            lines.append('| '+label+' | '+' | '.join(fmt(entries[k]) for k in ['psnr','fine_sam_miou','old_psnr','update_p95_ms'])+' |')
        delta={r['metric']:r for r in contrast if r['dataset']==ds and r['contrast']=='111_minus_000'}
        lines+=['',f'实测完整模型相对基线：PSNR {delta["psnr"]["mean"]:+.3f} dB，细区域一致性 {delta["fine_sam_miou"]["mean"]:+.4f}，旧视角末期 PSNR {delta["old_psnr"]["mean"]:+.3f} dB。这些差值按实测保留，不能预设三项都有效。','']
    lines+=['## 语义层级与计算开销','',
        f'独立教师基准：模型装载 {teacher["model_load_seconds"]:.2f} 秒；SAM+CLIP 的整卡采样峰值 {teacher["whole_gpu_sampled_peak_mib"]:.0f} MiB。教师每隔 6 个训练帧执行一次；该成本没有混入建图优化步耗时。','']
    for ds in summary:
        times=[r['seconds'] for r in teacher['frames'] if r['dataset']==ds]
        graph=provenance[ds]['prefix_preparation'];last=prefix[ds][-1]
        lines.append(f'- {ds}：教师单帧 {min(times):.2f}–{max(times):.2f} 秒（3 帧，含首帧预热）；前缀图准备 {graph["seconds"]:.2f} 秒；最终 {last["nodes"]} 个候选区域节点。可比较锚点的父子一致率 {last["candidate_parent_anchor_agreement"]:.1%}，这项仅检查内部包含关系，不证明部件语义正确。')
    lines+=['','PyTorch 优化循环峰值只统计张量分配，在每阶段点准入之后重置，因此不包含初始化/追加点时的瞬时峰值。`evaluation_gpu_peak_mib` 是该阶段优化与评估的共同累计峰值。`runs.csv` 另有 1 Hz nvidia-smi 整卡采样值（可漏掉短峰，第一组监测不完整）。进程总耗时含导入、CPU 数据读取、评估与导出；建图总耗时含逐前缀初始化/准入和训练，不含评估。控制器累计计时包含选图及数据访问，不单独分解增密策略耗时；增密开销包含在优化步中。','',
        '## 审计与边界','',
        f'- {n} 组均检查完成标记、每阶段步数/上限、实际优化像素预算和留出相机集合。全部检查点均通过有限数值与语义身份检查。冻结的训练源码哈希保持一致。',
        '- 另有未来 RGB/深度/位姿扰动不改变 TUM 已到达前缀锚点、留出深度不准入、相机裁剪射线等价、区域匹配计分和高斯分裂身份继承检查。',
        '- TUM 使用已知真值相机位姿；厨房使用离线 COLMAP 位姿、三角化点与全局归一化。厨房的点释放限制不能消除离线 SfM 的未来信息。因此是已知位姿下的六批增量建图，不是严格在线 RGB SLAM。',
        '- SAM 细区域是候选部件，CLIP 名称可能错；父子关系可能冲突，不能声称可靠命名部件树。',
        '- 厨房训练缓存复用了上一实验 41 个训练视角的 SAM/CLIP 输出（原图宽 779），掩码缩放到本次宽 640；新生成参考视角为 640。各组使用相同缓存。精确重现本次对比应使用归档教师/前缀文件。',
        '- 没有在线学习 SAGA 式亲和场；本版使用持久区域 ID 与有界 CLIP 多视角原型。几何 RGB/深度损失保持相同，语义通过控制器影响学习过程。',
        '- S 的开关同时控制语义新颖性/RGB 残差选图和区域裁剪；本次模块消融不能区分这三个子因素的独立贡献。',
        '- 区域渲染采用已知 ID 的透明度加权投票，未知高斯不投票；微弱背景投票可能延伸到未标注区域，覆盖率不能单独当作分割准确率。',
        '- Gaussian 及图像缓存/回放池有界，但当前实现预载数据集锚点，历史节点元信息仍会增长。不能声称总 CPU 内存对无限流恒定。',
        '- 两个场景不能代表多房间/动态环境。600 步/阶段验证有限计算预算下的表现，不是收敛质量；不与先前不同分辨率/步数的实验混作公平比较。','',
        '## 文件','',
        f'`analysis/runs.csv`：{n} 组原始汇总；`stages.csv`：{n*6} 个阶段；`contrasts.csv`：主效应、配对去模块与二阶交互。`results/` 保留每视角分数、训练日志、采样与增密审计。',
        f'归档含全部指标、不可变教师/前缀输入，以及两个场景 seed 7 的 000/111 四个完整检查点和导出地图。其余 {n-4} 个计划内检查点暂存原 Colab 运行时，未包含在便携结果包内。数据集原图请从官方来源重新取得。','',
        '来源：[TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)、[Mip-NeRF 360](https://jonbarron.info/mipnerf360/)、[gsplat](https://github.com/nerfstudio-project/gsplat)、[SAM 2](https://github.com/facebookresearch/sam2)、[SAGA](https://github.com/Jumpat/SegAnyGAussians)、[LaGa](https://github.com/SJTU-DeepVisionLab/LaGa)。','']
    (ROOT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines));print('EVIDENCE_BASED_REPORT_WRITTEN')

if __name__=='__main__':main()
