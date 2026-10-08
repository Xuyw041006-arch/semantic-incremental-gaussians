'use strict';
const $=id=>document.getElementById(id),data=window.PROJECT_DATA;
const modeNames={rgb:'Gaussian 外观渲染',objects:'物品区域分割 · SAM2 / CLIP 预测',fine:'细粒度区域候选 · 层级语义'};
let stage=4,mode='rgb',mapStarted=false,mapReady=false,trajectory=false,playing=false,toastTimer;
const video=$('inputVideo'),iframe=$('mapFrame');
function message(action,extra={}){if(mapReady&&iframe.contentWindow)iframe.contentWindow.postMessage({type:'gaussian-showcase',action,...extra},location.origin);}
function toast(text){$('toast').textContent=text;$('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),3200);}
function setHero(next){document.querySelectorAll('[data-hero-mode]').forEach(b=>{const active=b.dataset.heroMode===next;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});$('heroImage').src=`assets/stage-4-${next}.webp`;$('heroImage').alt=modeNames[next];$('heroCaption').textContent=modeNames[next];}
document.querySelectorAll('[data-hero-mode]').forEach(b=>b.addEventListener('click',()=>setHero(b.dataset.heroMode)));
function renderStage(next,seek=false){stage=next;const s=data.stages.find(s=>s.stage===next);$('mapWaiting').hidden=next!==0;$('mapStageBadge').textContent=next?'STAGE '+String(next).padStart(2,'0')+' / 04':'INITIALIZING';$('mapCountBadge').textContent=s?s.count.toLocaleString('en-US')+' GAUSSIANS':'等待第一批观测';$('pointCount').textContent=s?s.count.toLocaleString('en-US'):'—';$('updateTime').innerHTML=s?s.p95.toFixed(2)+' <small>ms</small>':'—';$('memoryUsage').innerHTML=s?Math.round(s.memory)+' <small>MiB</small>':'—';$('frameCount').innerHTML=s?s.end+' <small>帧</small>':'积累中';$('stageDescription').textContent=s?`阶段 ${next} · ${s.end} 帧观测 · ${s.count.toLocaleString('en-US')} 个高斯 / ${s.cap.toLocaleString('en-US')} 容量`:'正在积累第一批图像，尚未初始化地图';document.querySelectorAll('[data-stage]').forEach(b=>{b.classList.toggle('active',Number(b.dataset.stage)===next);b.setAttribute('aria-pressed',String(Number(b.dataset.stage)===next));});if(s){$('mapPoster').src=`assets/stage-${next}-${mode}.webp`;$('mapPoster').alt=`阶段 ${next}，${modeNames[mode]}`;}message('stage',{stage:next});if(seek&&s){if(video.readyState>0)video.currentTime=Math.min(s.time,video.duration||s.time);else video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(s.time,video.duration);},{once:true});}}
function setMode(next){mode=next;document.querySelectorAll('[data-demo-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.demoMode===next);b.setAttribute('aria-pressed',String(b.dataset.demoMode===next));});if(stage)$('mapPoster').src=`assets/stage-${stage}-${next}.webp`;message('mode',{mode:next});}
document.querySelectorAll('[data-demo-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.demoMode)));
document.querySelectorAll('[data-stage]').forEach(b=>b.addEventListener('click',()=>{stopPlayback();renderStage(Number(b.dataset.stage),true);}));
function startMap(){if(mapStarted)return;mapStarted=true;$('loadMap').disabled=true;$('loadMap').textContent='正在加载地图…';iframe.src='demo/viewer.html';iframe.hidden=false;$('mapPoster').hidden=true;$('loadPrompt').hidden=true;$('mapHint').hidden=false;}
$('loadMap').addEventListener('click',startMap);
window.addEventListener('message',event=>{if(event.source!==iframe.contentWindow||event.origin!==location.origin||event.data?.type!=='gaussian-showcase-ready')return;mapReady=true;message('stage',{stage});message('mode',{mode});message('trajectory',{value:trajectory});});
$('resetView').addEventListener('click',()=>{if(!mapStarted){startMap();toast('正在启动完整三维地图');}else message('reset');});
$('toggleTrajectory').addEventListener('click',()=>{trajectory=!trajectory;$('toggleTrajectory').setAttribute('aria-pressed',String(trajectory));$('toggleTrajectory').textContent=trajectory?'隐藏相机轨迹':'显示相机轨迹';if(!mapStarted)startMap();else message('trajectory',{value:trajectory});});
function stopPlayback(){playing=false;video.pause();$('playReconstruction').textContent='▶ 播放建图过程';}
$('playReconstruction').addEventListener('click',async()=>{if(playing){stopPlayback();return;}playing=true;if(!mapStarted)startMap();video.currentTime=0;renderStage(0);$('playReconstruction').textContent='Ⅱ 暂停建图回放';try{await video.play();}catch{stopPlayback();toast('视频暂未加载，请稍后再试');}});
video.addEventListener('timeupdate',()=>{if(!playing)return;const t=video.currentTime,available=data.stages.filter(s=>s.time<=t+.06),next=available.at(-1)?.stage||0;if(next!==stage)renderStage(next);const arrived=data.frameTimes.filter(x=>x<=t+.06).length;$('frameCount').innerHTML=arrived+' <small>帧</small>';});
video.addEventListener('ended',()=>{stopPlayback();renderStage(4,false);});
video.addEventListener('error',()=>toast('视频读取失败，请通过 HTTP 服务打开页面'));
const steps=[
['把连续视频变成有效的多视角观测。','支持 PyAV 可解码的 MP4、MOV、MKV、AVI 等视频。按时间抽帧、过滤重复画面、处理旋转，并保留时间戳，建立后续重建的统一数据入口。','PyAV · 时间抽帧 · 重复帧过滤'],
['从 RGB 图像恢复相机与稀疏几何。','使用 COLMAP 估计相机内参、位姿和稀疏点云，对图像去畸变，并检查定位率与重投影误差。相机估计采用完整视频的离线 SfM，为后续增量建图提供几何基础。','COLMAP · 相机位姿 · 稀疏点云 · 去畸变'],
['分出区域，也赋予区域可匹配的描述。','SAM2 产生物品与细区域候选，CLIP 提供区域语义描述。通过任务文字查询定义重要物品，让后续重建可以关注具体目标与轮廓。','SAM2 · CLIP · 任务重要性 · 区域置信度'],
['把不同画面里的区域，关联到同一地图。','按已经到达的训练图像前缀，结合三维重叠和 CLIP 描述关联跨视角区域。保留物品级与细区域级 ID、置信度和观测记录，为分层语义控制提供依据。','三维重叠 · 跨视角关联 · 两级语义节点'],
['继承已有地图，更新需要补全的地方。','新一批图像到达后，更新已有高斯参数并按预算释放新点。语义选图、重要物品与边界增密、覆盖回放共同控制训练；新生高斯通过成熟条件后再参与增密。','PyTorch / CUDA · gsplat · S / B / R 控制器'],
['交互探索场景，导出可复查的结果。','浏览器显示每阶段的完整各向异性 Gaussian 地图、相机轨迹与计算开销，支持物品及细区域切换。输出标准 3DGS PLY、语义标签、模型参数、Adam 检查点和原始指标。','WebGL2 · 完整 splats · PLY · 检查点与审计']
];
const stepButtons=[...document.querySelectorAll('[data-step]')];
function selectStep(i,focus=false){stepButtons.forEach((b,j)=>{b.classList.toggle('active',j===i);b.setAttribute('aria-selected',String(j===i));b.tabIndex=j===i?0:-1;});$('pipelineNumber').textContent=String(i+1).padStart(2,'0');$('pipelineTitle').textContent=steps[i][0];$('pipelineText').textContent=steps[i][1];$('pipelineTech').textContent=steps[i][2];if(focus)stepButtons[i].focus();}
stepButtons.forEach((b,i)=>{b.addEventListener('click',()=>selectStep(i));b.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?5:(i+(e.key==='ArrowRight'?1:5))%6;selectStep(next,true);});});
$('compareRange').addEventListener('input',e=>{const value=Number(e.target.value);$('compareOriginal').style.clipPath=`inset(0 ${100-value}% 0 0)`;$('compareHandle').style.left=value+'%';});
selectStep(0);renderStage(4,true);
if(location.protocol==='file:')toast('完整三维交互请通过本地 HTTP 服务或部署地址打开');
