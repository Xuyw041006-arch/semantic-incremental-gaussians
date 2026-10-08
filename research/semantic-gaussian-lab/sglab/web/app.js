"use strict";
const $=id=>document.getElementById(id);
let labels=[],config={},length=100,nextFrame=0,map=[],history=[],trajectory=[],objects=[],images=null,camera=null;
let replay=false,viewHint=null;
let running=false,busy=false,mode="rgb",source="rgb",filterLabel=null,filterInstance=null,timer=null;
let yaw=.6,pitch=.65,distance=9,target=[0,.65,0],dirty=true;
const canvas=$("scene"),gl=canvas.getContext("webgl",{alpha:true,antialias:true});
function instanceColor(id){return id>0?[70+(id*53)%170,70+(id*97)%170,70+(id*131)%170]:[78,88,101];}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
let program,buffer,locations={};
if(gl){
  program=gl.createProgram();
  gl.attachShader(program,shader(gl.VERTEX_SHADER,`
    attribute vec3 aPosition; attribute vec3 aColor; attribute float aSigma; attribute float aOpacity;
    uniform vec3 uEye; uniform vec3 uRight; uniform vec3 uUp; uniform vec3 uForward;
    uniform float uAspect; uniform float uHeight; uniform float uPoint;
    varying vec3 vColor; varying float vOpacity;
    void main(){vec3 p=aPosition-uEye;float z=dot(uForward,p);float x=dot(uRight,p);float y=dot(uUp,p);
    gl_Position=vec4(x/(uAspect*0.48),y/0.48,1.002*z-0.1001,z);
    gl_PointSize=clamp(aSigma*5.0*uHeight/(0.96*max(z,0.05)),1.0,96.0);
    vColor=aColor;vOpacity=aOpacity;}`));
  gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`
    precision mediump float;varying vec3 vColor;varying float vOpacity;uniform float uPoint;
    void main(){float alpha=vOpacity;if(uPoint>0.5){vec2 p=(gl_PointCoord-0.5)*5.0;
    float r=dot(p,p);if(r>6.25)discard;alpha*=exp(-0.5*r);if(alpha<0.015)discard;}
    gl_FragColor=vec4(vColor,alpha);}`));
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);buffer=gl.createBuffer();
  for(const name of ["aPosition","aColor","aSigma","aOpacity"])locations[name]=gl.getAttribLocation(program,name);
  for(const name of ["uEye","uRight","uUp","uForward","uAspect","uHeight","uPoint"])locations[name]=gl.getUniformLocation(program,name);
}else{$("mapError").hidden=false;$("mapError").textContent="此浏览器未启用 WebGL。请在支持 WebGL 的 Chrome / Safari 中打开本地地址。";}

const normalize=v=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
function vertices(points,colors,sigmas=null,opacities=null){
  const a=new Float32Array(points.length*8);
  points.forEach((p,i)=>{a.set(p,i*8);a.set(colors[i].map(v=>v/255),i*8+3);a[i*8+6]=sigmas?sigmas[i]:.02;a[i*8+7]=opacities?opacities[i]:1;});return a;
}
function drawArray(array,kind,point){
  if(!array.length)return;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,array,gl.DYNAMIC_DRAW);
  for(const [name,size,offset] of [["aPosition",3,0],["aColor",3,12],["aSigma",1,24],["aOpacity",1,28]]){gl.enableVertexAttribArray(locations[name]);gl.vertexAttribPointer(locations[name],size,gl.FLOAT,false,32,offset);}
  gl.uniform1f(locations.uPoint,point);gl.drawArrays(kind,0,array.length/8);
}
function render(){
  if(!gl||!dirty)return;dirty=false;
  const ratio=Math.min(devicePixelRatio||1,2),width=Math.round(canvas.clientWidth*ratio),height=Math.round(canvas.clientHeight*ratio);
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  gl.viewport(0,0,width,height);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  const eye=[target[0]+distance*Math.cos(pitch)*Math.sin(yaw),target[1]+distance*Math.sin(pitch),target[2]+distance*Math.cos(pitch)*Math.cos(yaw)];
  const forward=normalize(target.map((v,i)=>v-eye[i])),right=normalize(cross(forward,[0,1,0])),up=cross(right,forward);
  gl.useProgram(program);gl.uniform3fv(locations.uEye,eye);gl.uniform3fv(locations.uForward,forward);gl.uniform3fv(locations.uRight,right);gl.uniform3fv(locations.uUp,up);
  gl.uniform1f(locations.uAspect,width/height);gl.uniform1f(locations.uHeight,height);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);
  const grid=[],gridColors=[];
  for(let i=-3;i<=3;i+=.5){grid.push([-3,-.02,i],[3,-.02,i],[i,-.02,-3],[i,-.02,3]);gridColors.push(...Array(4).fill([43,59,69]));}
  drawArray(vertices(grid,gridColors),gl.LINES,0);
  const visible=map.filter(p=>p&&(filterLabel===null||p.semantic===filterLabel)&&(filterInstance===null||p.instance===filterInstance));
  visible.sort((a,b)=>dot(b.xyz,forward)-dot(a.xyz,forward));
  const points=visible.map(p=>p.xyz),colors=visible.map(p=>mode==="rgb"?p.rgb:mode==="semantic"?(labels[p.semantic]?.color||[113,123,137]):instanceColor(p.instance));
  drawArray(vertices(points,colors,visible.map(p=>p.sigma),visible.map(p=>p.opacity??.9)),gl.POINTS,1);
  if(trajectory.length>1)drawArray(vertices(trajectory,trajectory.map(()=>[156,232,193])),gl.LINE_STRIP,0);
  if(camera){
    const transform=p=>[0,1,2].map(i=>camera[i][3]+p.reduce((s,v,j)=>s+camera[i][j]*v,0));
    const origin=transform([0,0,0]),corners=[[-.13,-.1,.22],[.13,-.1,.22],[.13,.1,.22],[-.13,.1,.22]].map(transform),lines=[];
    corners.forEach((p,i)=>lines.push(origin,p,p,corners[(i+1)%4]));
    drawArray(vertices(lines,lines.map(()=>[229,240,229])),gl.LINES,0);
  }
}
function updateMap(packet,full){if(full)map=[];packet.ids.forEach((id,i)=>{map[id]={xyz:packet.xyz.slice(i*3,i*3+3),rgb:packet.rgb.slice(i*3,i*3+3),sigma:packet.sigma[i],semantic:packet.semantic[i],instance:packet.instance[i],opacity:packet.opacity?.[i]??.9};});dirty=true;}
function chart(id,key,ceiling,color){const c=$(id),ctx=c.getContext("2d"),r=devicePixelRatio||1;c.width=c.clientWidth*r;c.height=c.clientHeight*r;ctx.scale(r,r);const w=c.clientWidth,h=c.clientHeight;const values=history.map(s=>s[key]);if(!values.length)return;const max=Math.max(ceiling,...values,1);ctx.beginPath();values.forEach((v,i)=>{const x=i/Math.max(1,values.length-1)*w,y=h-2-v/max*(h-4);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.stroke();if(key==="update_ms"){ctx.setLineDash([3,4]);ctx.beginPath();const y=h-2-ceiling/max*(h-4);ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.strokeStyle="#657487";ctx.lineWidth=.7;ctx.stroke();}}
function updateImages(){if(images){$("inputImage").src=images[source];$("inputImage").style.display="block";$("sourcePlaceholder").hidden=true;}else{$("inputImage").style.display="none";$("sourcePlaceholder").hidden=false;}$("sourceTag").textContent={rgb:"RGB",semantic:"SEMANTIC",instance:"FRAME-LOCAL INSTANCE"}[source];}
function updateUI(){
  const s=history.at(-1);$("frameNumber").textContent=`${String(nextFrame).padStart(3,"0")} / ${length}`;
  $("progressFill").style.width=`${nextFrame/length*100}%`;$("progressText").textContent=`${(s?.timestamp??0).toFixed(1)} s · ${nextFrame}/${length} 帧`;
  $("gaussianCount").innerHTML=`${(s?.gaussians??0).toLocaleString()} <small>/ ${config.max_gaussians.toLocaleString()}</small>`;
  $("updateMs").innerHTML=s?`${(s.mapping_total_ms??s.update_ms).toFixed(1)} <small>ms${s.budget_overrun?" · 超软预算":""}</small>`:"— <small>ms</small>";
  $("mapMemory").innerHTML=`${(s?.map_used_mib??0).toFixed(2)} <small>MiB 已用</small>`;
  $("capacity").textContent=`数组容量 ${s?.map_capacity_mib??"—"} MiB · 不含 Python 索引`;
  const peaks=history.map(x=>x.gpu_peak_mib??0); const peak=Math.max(0,...peaks);
  $("gpuMemory").innerHTML=peak?`${peak.toFixed(0)} <small>MiB 峰值</small>`:"未测量";
  $("rss").textContent=`CPU 进程峰值 RSS ${s?.process_peak_rss_mib??"—"} MiB`;
  $("mapStatus").textContent=s?`+${s.added} 新增 · ${s.stable} 稳定 · ${s.deferred} 延后`:"等待第一帧";
  $("objectCount").textContent=`${objects.length} 个轨迹 ID`;
  $("objects").replaceChildren();
  if(!objects.length){const p=document.createElement("p");p.className="empty";p.textContent="观察到的物品会在这里获得跨帧 ID";$("objects").append(p);}
  objects.forEach(o=>{const row=document.createElement("div");row.className="object";row.tabIndex=0;row.setAttribute("role","button");const swatch=document.createElement("i");swatch.style.background=`rgb(${instanceColor(o.id)})`;const name=document.createElement("span");name.textContent=`${labels[o.label]?.zh??"物品"} #${o.id}`;const detail=document.createElement("small");detail.textContent=`${o.observations} 帧`;row.append(swatch,name,detail);const choose=()=>{filterInstance=filterInstance===o.id?null:o.id;filterLabel=null;$("selection").textContent=filterInstance===null?"全部类别":`${name.textContent} · 点击再次取消`;dirty=true;render();};row.onclick=choose;row.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose();}};$("objects").append(row);});
  $("step").disabled=busy||running||nextFrame>=length;$("reset").disabled=busy;$("export").disabled=busy;
  $("play").textContent=running?"Ⅱ 暂停":nextFrame>=length?"↺ 从头播放":"▶ 开始播放";
  if(replay&&!s){$("sourceCaption").textContent="TUM 真实 RGB-D · Mask2Former 预计算语义与实例分割";}
  const provenance=s?.segmentation_source;
  $("sourceCaption").textContent=replay&&!s?"TUM 真实 RGB-D · Mask2Former 预计算语义与实例分割":provenance==="mask2former_prediction"?`Mask2Former 预计算分割${s.cached_segmentation_ms?` · 推理 ${s.cached_segmentation_ms.toFixed(0)} ms` : ""}；2D 实例 ID 为帧内 ID。`:provenance==="none"?"未提供分割；未知类别不会自动识别为物品。":provenance==="provided"?"来自数据集的分割输入；2D 为帧内 ID，3D 为关联后 ID。":"合成输入的分割来自真值；尚未运行学习模型。";
  updateImages();chart("countChart","gaussians",config.max_gaussians,"#9ce8c1");chart("timeChart",replay?"mapping_total_ms":"update_ms",config.target_update_ms,"#aabedd");render();
}
async function api(path,payload){const options=payload===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)};const response=await fetch(path,options);const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);return result;}
function applyState(s){labels=s.labels;config=s.config;length=s.length;nextFrame=s.next_frame;history=s.history;trajectory=s.trajectory;objects=s.objects;images=s.images;camera=s.camera??null;replay=!!s.replay;viewHint=s.view_hint??null;
if(viewHint){target=viewHint.target;distance=viewHint.distance;pitch=.4;}
$("backendName").textContent=replay?"Colab T4 · CUDA":"CPU 参考后端";
$("backendNote").textContent=replay?"真实优化地图快照回放 · GT 位姿":"RGB-D 融合 · 已知相机位姿";
$("updateTitle").textContent=replay?"RECORDED FUSION + CUDA UPDATE":"CPU MAP UPDATE";
$("timingNote").textContent=replay?"开销来自 Colab 实测；分割预计算并单独计时；显存为 PyTorch 张量峰值；回放与图像编码不计入更新时间。当前为单独记录运行，基准计时见实验报告。":"时间来自实际逐帧计算；图表不含网络、图像编码和渲染。显存需要在 Colab CUDA 实验中测量。";
document.querySelector(".budget-panel").style.display=replay?"none":"";$("cap").disabled=replay;$("budget").disabled=replay;$("reset").textContent=replay?"回到开头":"重建";
updateMap(s.map,true);$("description").textContent=s.description;$("legend").replaceChildren();labels.forEach((label,i)=>{if(i===0||(s.visible_labels&&!s.visible_labels.includes(i)))return;const b=document.createElement("button");b.setAttribute("aria-pressed","false");const swatch=document.createElement("i");swatch.style.background=`rgb(${label.color})`;b.append(swatch,document.createTextNode(label.zh||label.name));b.onclick=()=>{filterLabel=filterLabel===i?null:i;filterInstance=null;document.querySelectorAll("#legend button").forEach(x=>x.setAttribute("aria-pressed","false"));b.setAttribute("aria-pressed",String(filterLabel!==null));$("selection").textContent=filterLabel===null?"全部类别":`${label.zh||label.name} · 点击再次取消`;dirty=true;render();};$("legend").append(b);});updateUI();}
function stop(){running=false;clearTimeout(timer);updateUI();}
async function step(){
  if(busy||nextFrame>=length)return;busy=true;updateUI();const start=performance.now();
  try{const s=await api("/api/step",{});if(s.stats){if(s.history)history=s.history;else history.push(s.stats);nextFrame=s.next_frame;trajectory=s.trajectory;objects=s.objects;images=s.images;camera=s.camera;updateMap(s.map,s.full);$("status").textContent=`帧 ${nextFrame} · 数据读取 ${s.stats.source_ms.toFixed(1)} ms · 请求 ${(performance.now()-start).toFixed(0)} ms`;}
    if(s.done)running=false;
  }catch(e){running=false;$("status").textContent=`错误：${e.message}`;}
  finally{busy=false;updateUI();}
  if(running)timer=setTimeout(step,Math.max(0,1000/Number($("speed").value)-(performance.now()-start)));
}
async function reset(){stop();busy=true;updateUI();try{const s=await api("/api/reset",{max_gaussians:Number($("cap").value),target_update_ms:Number($("budget").value)});filterLabel=null;filterInstance=null;$("selection").textContent="全部类别";applyState(s);$("status").textContent="已回到序列开头";}catch(e){$("status").textContent=e.message;}finally{busy=false;updateUI();}}
$("play").onclick=async()=>{if(running){stop();return;}if(nextFrame>=length)await reset();running=true;updateUI();step();};$("step").onclick=step;$("reset").onclick=reset;
$("export").onclick=async()=>{try{const result=await api("/api/export",{});$("status").textContent=`已导出 ${result.gaussians} 个 Gaussian：${result.path}`;}catch(e){$("status").textContent=e.message;}};
$("modes").onclick=e=>{const b=e.target.closest("button");if(!b)return;mode=b.dataset.mode;$("modes").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));dirty=true;render();};
$("sources").onclick=e=>{const b=e.target.closest("button");if(!b)return;source=b.dataset.source;$("sources").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));updateImages();};
$("home").onclick=()=>{yaw=.6;pitch=viewHint ? .4 : .65;distance=viewHint?.distance??9;target=viewHint?.target??[0,.65,0];dirty=true;render();};
let pointer=null;canvas.onpointerdown=e=>{pointer=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};canvas.onpointerup=()=>pointer=null;canvas.onpointercancel=()=>pointer=null;
canvas.onpointermove=e=>{if(!pointer)return;yaw-=(e.clientX-pointer[0])*.006;pitch=Math.max(-.1,Math.min(1.48,pitch+(e.clientY-pointer[1])*.006));pointer=[e.clientX,e.clientY];dirty=true;render();};
canvas.addEventListener("wheel",e=>{e.preventDefault();distance=Math.max(2,Math.min(18,distance*Math.exp(e.deltaY*.001)));dirty=true;render();},{passive:false});
new ResizeObserver(()=>{dirty=true;render();if(history.length)updateUI();}).observe(canvas);
api("/api/state").then(s=>{applyState(s);$("status").textContent="已连接本地引擎";}).catch(e=>{$("status").textContent=`连接失败：${e.message}`;});
