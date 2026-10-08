"""Data-driven Chinese results summary with actual exported comparison panels."""
from pathlib import Path
import json,html,base64,numpy as np
ROOT=Path.cwd()

def read(name):return json.loads((ROOT/name).read_text())
def main():
 summary=read('analysis/summary.json');contrasts=read('analysis/contrasts.json');verification=read('analysis/verification-runs.json');independent=read('analysis/verification-summary.json')
 names={'000':'基线','001':'R','010':'B','011':'B+R','100':'S','101':'S+R','110':'S+B','111':'S+B+R'}
 sections=[]
 def add(text):sections.append(text)
 def fmt(v,precision=3):return f'{v["mean"]:.{precision}f} ± {v["sd"]:.{precision}f}'
 def figure(path,caption):
  p=ROOT/path
  if p.exists():add('<figure><img src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
 add('<h1>增量式语义 Gaussian：实测结果汇总</h1>')
 gains={d:v['111']['psnr']['mean']-v['000']['psnr']['mean'] for d,v in summary.items()}
 add('<p><strong>完整模型相对基线：'+ '；'.join(html.escape(d)+f' {v:+.3f} dB' for d,v in gains.items())+'；未用于调参的 desk2 '+f'{independent["means"]["psnr"]:+.3f}'+' dB。</strong>开发场景的旧视角质量有提升；独立录像的旧视角两种子方向不一致，细区域分数平均基本持平。</p>')
 add('<p>三项控制器已经实现并完成消融，但没有证据证明三项都有效。当前主要收益来自 R；两个开发场景的 S+R 平均 PSNR 均略高于三项全开，B 尚未显示稳定收益。S+R 没有在独立录像上做额外比较，不能据此宣称它在独立录像也更好。下面保留全部正负结果，作为后续研究依据。</p>')
 add('<p>先完成 T4 上的 32 组完整消融，再根据负面结果改进控制器。在 L4 上重新训练基线与全部组合，共 32 组；另用 desk2 的 4 组独立录像实验验证，并补 4 组匹配回放频率的诊断对照，共 40 组。每组 6 阶段 × 600 步、宽 640、相同 RGB/深度损失、高斯上限逐步升至 30 万。两种种子 7/17，误差条为样本标准差，不作显著性声明。</p>')
 add('<h2>改进了什么</h2><ul><li>S：一半保留均匀探索，一半按 RGB 难度与温和语义新颖性选图；减少过强的局部裁剪。</li><li>B：语义稀缺度对点准入、增密、置换的权重有上限，让重建误差继续主导容量使用。</li><li>R：回放池加强时间覆盖，旧帧采样概率从 0.3 逐步升至最后阶段 0.55，减轻旧区域遗忘。</li></ul>')
 add('<p>第一版两个场景的 B 平均 PSNR 效应均为负；完整组合在 TUM 略有下降，因此没有把第一版当成成功结果。所有第一版指标保留在独立归档中。原协议 3 种种子在查看汇总前按用户要求改为前两种完整种子，修订记录可查。</p>')
 add('<h2>同一 L4 上的公平对照</h2>')
 for ds,v in summary.items():
  add('<h3>'+html.escape(ds)+'</h3><table><tr><th>组合</th><th>PSNR dB</th><th>SSIM</th><th>物体区域一致性</th><th>细区域一致性</th><th>小细区域</th><th>旧视角末期 PSNR</th></tr>')
  for f,x in v.items():add('<tr><td>'+names[f]+' '+f+'</td>'+''.join('<td>'+fmt(x[k])+'</td>' for k in ['psnr','ssim','object_sam_miou','fine_sam_miou','small_fine_sam_miou','old_psnr'])+'</tr>')
  add('</table><p>完整模型 − 基线：'+ '；'.join(label+f' {v["111"][key]["mean"]-v["000"][key]["mean"]:+.4f}' for label,key in [('PSNR dB','psnr'),('细区域一致性','fine_sam_miou'),('旧视角 PSNR dB','old_psnr')])+'。</p>')
  add('<table><tr><th>组合</th><th>实际点数</th><th>建图秒</th><th>最末阶段 p95 ms</th><th>整卡采样峰值 MiB</th><th>PyTorch 优化循环峰值 MiB</th></tr>')
  for f in ['000','111']:
   x=v[f];add('<tr><td>'+names[f]+'</td>'+''.join('<td>'+fmt(x[k],1)+'</td>' for k in ['gaussians','mapping_seconds','update_p95_ms','whole_gpu_sampled_peak_mib','gpu_update_peak_mib'])+'</tr>')
  add('</table><h4>配对去模块消融</h4><table><tr><th>完整模型减去</th><th>PSNR Δ</th><th>细区域 Δ</th><th>旧视角 PSNR Δ</th></tr>')
  for tag,label in [('111_minus_011_S','关闭 S'),('111_minus_101_B','关闭 B'),('111_minus_110_R','关闭 R')]:
   cs={x['metric']:x for x in contrasts if x['dataset']==ds and x['contrast']==tag}
   add('<tr><td>'+label+'</td>'+''.join('<td>'+fmt(cs[k])+'</td>' for k in ['psnr','fine_sam_miou','old_psnr'])+'</tr>')
  add('</table><p>正值表示开启该模块有帮助；负值仍如实保留。完整 2×2×2 主效应与交互项见 contrasts.csv。</p>')
  figure(f'analysis/{ds}-factorial.png','八种开关组合的实测比较，种子为重复单位。')
  figure(f'analysis/{ds}-stages.png','同一批早期留出相机的质量曲线，以及实际高斯数量。')
  figure(f'analysis/{ds}-visual-early.png','真实留出图像与最终阶段重建；颜色表示持久区域身份。')
 control=read('analysis/replay-control-runs.json');control_deltas=read('analysis/replay-control-summary.json')['paired_semantic_R_minus_time_pose_control']
 add('<h2>语义回放还是更多回放？</h2><p>这是主消融之后追加的诊断，不用于重新筛选候选模型。只开启 R，保留完全相同的回放概率、时间探索、相机位置新颖性和容量，把语义覆盖信息清空。下表的差值为语义 R 减去该时间/位置对照；正值才支持语义区域选择有额外帮助。</p><table><tr><th>场景</th><th>时间/位置对照 PSNR</th><th>语义选择 PSNR Δ</th><th>旧视角 PSNR Δ</th><th>细区域一致性 Δ</th></tr>')
 for ds in ['tum-desk','kitchen']:
  rs=[x for x in control if x['dataset']==ds];ps=[x['delta'] for x in control_deltas if x['dataset']==ds]
  values=[[x['psnr'] for x in rs]]+[[x[k] for x in ps] for k in ['psnr','old_psnr','fine_sam_miou']]
  add('<tr><td>'+ds+'</td>'+''.join('<td>'+fmt(dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1))))+'</td>' for v in values)+'</tr>')
 add('</table><p>R 的八组合消融效应同时包含回放概率与选帧方式，不能全部归因为语义；该额外对照只隔离 R 的语义选择，不隔离 S 的语义成分与 RGB 难度成分。</p>')
 add('<h2>未用于调参的 desk2 录像</h2><p>它是同一办公室的另一段拍摄，不代表跨环境泛化。方案在读取本序列训练结果前锁定。独立结果无论正负都保留。</p><table><tr><th>组合</th><th>PSNR dB</th><th>细区域一致性</th><th>旧视角末期 PSNR dB</th></tr>')
 for f in ['000','111']:
  subset=[r for r in verification if r['factors']==f]
  add('<tr><td>'+names[f]+'</td>'+''.join('<td>'+fmt(dict(mean=float(np.mean([r[k] for r in subset])),sd=float(np.std([r[k] for r in subset],ddof=1))))+'</td>' for k in ['psnr','fine_sam_miou','old_psnr'])+'</tr>')
 add('</table><p>配对平均差：'+ '；'.join(k+f' {v:+.4f}' for k,v in independent['means'].items())+'。</p>')
 figure('analysis/tum-desk2-visual-early.png','独立录像的实际重建与分割输出。')
 depth=read('analysis/depth-summary.json')
 add('<h2>留出深度检查</h2><p>仅对有米制深度的两个 TUM 录像，在训练结束后额外检查最终地图。它不是筛选方案的指标。覆盖表面误差和覆盖率同时报告；另将未覆盖像素赋为 5 米计算带缺失惩罚的 RMSE，避免只保留易重建像素。</p><table><tr><th>录像</th><th>组合</th><th>覆盖表面 RMSE m</th><th>有效深度覆盖率</th><th>带缺失惩罚 RMSE m</th></tr>')
 for ds in ['tum-desk','tum-desk2']:
  for flag in ['000','111']:
   rows=[x for x in depth if x['dataset']==ds and x['factors']==flag]
   add('<tr><td>'+ds+'</td><td>'+names[flag]+'</td>'+''.join('<td>'+f'{np.mean([x[k] for x in rows]):.4f}'+'</td>' for k in ['covered_depth_rmse_m','depth_coverage','penalized_depth_rmse_m'])+'</tr>')
 add('</table>')
 teacher=read('teacher-benchmark.json')
 add('<h2>独立语义教师开销</h2><p>L4 上单独重测 SAM+CLIP：模型装载 '+f'{teacher["model_load_seconds"]:.2f}'+' 秒，整卡 0.2 秒采样峰值 '+f'{teacher["whole_gpu_sampled_peak_mib"]:.0f}'+' MiB。每个开发场景取 3 个训练帧；单帧耗时包含读取、SAM 与 CLIP，首帧含预热。</p><ul>')
 for ds in ['tum-desk','kitchen']:
  t=[r['seconds'] for r in teacher['frames'] if r['dataset']==ds]
  add('<li>'+ds+f'：{min(t):.2f}–{max(t):.2f} 秒/教师帧。</li>')
 add('</ul><p>教师每隔 6 个训练帧使用一次，各组合共享缓存。该开销未计入优化步或建图秒数，不能仅凭建图 p95 声称端到端实时。</p>')
 add('<h2>如何理解这些结果</h2><p>新画面到达时，系统保留现有高斯与优化器状态，逐批加入新锚点，在有限步数内继续更新。语义在这里主要负责选择学习位置、分配点数和保留旧区域；不是重新从零训练整张地图。物体层与细区域层分别维护持久 ID，CLIP 提供候选名称。</p>')
 add('<p>SAM 是区域参考，细区域是候选部件；这里的匹配一致性不是人工语义类别或部件真值 mIoU。已知 ID 的透明度投票中未知高斯不投票，弱背景投票仍可能扩散，不能把覆盖率当成准确率。</p>')
 add('<p>TUM 使用已知真值相机位姿，厨房使用离线 COLMAP 位姿与点云，后者可能包含未来或留出视角信息。此实验验证已知位姿下的六批增量建图，不是完整相机跟踪 SLAM。当前没有在线学习 SAGA 式亲和特征场，B 使用覆盖代理分数，未实现高斯合并。</p>')
 add('<p>点数、回放池与图像缓存有界；历史元信息和预载锚点仍随数据增长。建图时间不含语义教师、评估和导出；语义教师缓存为各组合共享。更新 p95 来自最末阶段，不能视为端到端实时保证。PyTorch 峰值在点准入后重置，遗漏初始化瞬时峰；整卡数值为 1 Hz 采样，也可能漏掉短峰。</p>')
 add('<p>筛查和迭代使用原两个场景的 seed 7，seed 17 用于确认。原两场景在第一版后已成为开发数据，不把反复查看其结果说成独立测试。独立 desk2 结果没有用于筛选此版本。两种种子及三段有限序列不足以证明普遍优势。</p>')
 add('<h2>原始证据和复现</h2><p>analysis/runs.csv、stages.csv、contrasts.csv 保存主实验与配对统计；verification-runs.csv 保存独立录像数据。results 下有逐视角指标、采样审计、增密日志。源码哈希、每阶段状态和环境版本随归档提供。归档保留三个场景 seed 7 完整模型的检查点与已导出地图；原图需从官方数据集重新下载。</p>')
 add('<p>第一版便携包保留四个代表性检查点；其余第一版检查点没有包含在备份中，更换运行时后不再作为可恢复交付物。原始指标与日志已保存。不同 GPU 的耗时不混作方法加速收益。</p>')
 add('<p>数据与基础工具：<a href="https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download">TUM RGB-D</a> · <a href="https://jonbarron.info/mipnerf360/">Mip-NeRF 360</a> · <a href="https://github.com/nerfstudio-project/gsplat">gsplat</a> · <a href="https://github.com/facebookresearch/sam2">SAM 2</a></p>')
 css='body{font:16px/1.65 system-ui,sans-serif;color:#25313a;background:#f4f6f7;margin:0}main{max-width:1200px;margin:auto;background:white;padding:35px}h1,h2,h3{color:#17394b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d8e0e5;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf3f6}figure{margin:26px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#5c6b75}p{max-width:1050px}'
 (ROOT/'RESULTS.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>增量语义 Gaussian 实验结果</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>')
 print('ACTUAL_RESULTS_HTML_WRITTEN')
if __name__=='__main__':main()
