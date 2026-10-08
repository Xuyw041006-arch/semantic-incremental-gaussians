"use strict";
const $=id=>document.getElementById(id), canvas=$("scene"),gl=canvas.getContext("webgl2",{alpha:false,antialias:true});
let meta=null,points=null,loadedStage=0,frame=0,mode=-1,playing=false,timer=null,request=0,queryRequest=0,selection=null,seed=null;
let follow=true,yaw=.3,pitch=.4,distance=1.8,eye=[],right=[],up=[],forward=[],focal=1.8;
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=v=>{const n=Math.hypot(...v)||1;return v.map(x=>x/n)},cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function program(v,f){const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,v));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,f));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
const vertex=`#version 300 es
precision highp float;layout(location=0) in vec2 aCorner;layout(location=1) in vec3 aPosition;layout(location=2) in vec3 aScale;layout(location=3) in vec4 aQuat;layout(location=4) in vec4 aColor;
uniform vec3 uEye,uRight,uUp,uForward;uniform vec2 uSize;uniform float uFocal;out vec2 vLocal;out vec4 vColor;
void main(){vec3 p=aPosition-uEye;float x=dot(p,uRight),y=dot(p,uUp),z=dot(p,uForward);vColor=aColor;vLocal=aCorner*3.0;
if(z<.01){gl_Position=vec4(3,3,3,1);vColor.a=0.0;return;}
vec4 q=normalize(aQuat);float w=q.x,rx=q.y,ry=q.z,rz=q.w;
mat3 R=mat3(1.0-2.0*(ry*ry+rz*rz),2.0*(rx*ry+w*rz),2.0*(rx*rz-w*ry),2.0*(rx*ry-w*rz),1.0-2.0*(rx*rx+rz*rz),2.0*(ry*rz+w*rx),2.0*(rx*rz+w*ry),2.0*(ry*rz-w*rx),1.0-2.0*(rx*rx+ry*ry));
float fp=uFocal*uSize.y*.5;mat2 C=mat2(.3,0,0,.3);
for(int k=0;k<3;k++){vec3 a=R[k]*aScale[k];vec2 d=fp*vec2(dot(a,uRight)*z-x*dot(a,uForward),dot(a,uUp)*z-y*dot(a,uForward))/(z*z);C+=outerProduct(d,d);}
float trace=C[0][0]+C[1][1],delta=sqrt(max(0.0,(C[0][0]-C[1][1])*(C[0][0]-C[1][1])+4.0*C[0][1]*C[0][1]));float l1=max(.3,(trace+delta)*.5),l2=max(.3,(trace-delta)*.5);
vec2 axis=abs(C[0][1])>.0001?normalize(vec2(C[0][1],l1-C[0][0])):vec2(1,0);vec2 axis2=vec2(-axis.y,axis.x);
vec2 offset=axis*min(100.0,sqrt(l1))*vLocal.x+axis2*min(100.0,sqrt(l2))*vLocal.y;
vec2 center=vec2(x*uFocal*uSize.y/uSize.x/z,y*uFocal/z);gl_Position=vec4(center+offset*2.0/uSize,0,1);}`;
const fragment=`#version 300 es
precision highp float;in vec2 vLocal;in vec4 vColor;out vec4 color;void main(){float r=dot(vLocal,vLocal);if(r>9.0)discard;float a=vColor.a*exp(-.5*r);if(a<.01)discard;color=vec4(vColor.rgb,a);}`;
let prog,vao,instanceBuffer,lineProg,lineBuffer;
if(gl){
  prog=program(vertex,fragment);vao=gl.createVertexArray();gl.bindVertexArray(vao);
  const corners=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,corners);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  instanceBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,instanceBuffer);
  for(const [loc,size,off] of [[1,3,0],[2,3,12],[3,4,24],[4,4,40]]){gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,68,off);gl.vertexAttribDivisor(loc,1);}
  lineProg=program(`#version 300 es\nprecision highp float;layout(location=0) in vec3 p;uniform vec3 uEye,uRight,uUp,uForward;uniform vec2 uSize;uniform float uFocal;void main(){vec3 d=p-uEye;float z=dot(d,uForward);if(z<.01){gl_Position=vec4(3,3,3,1);return;}gl_Position=vec4(dot(d,uRight)*uFocal*uSize.y/uSize.x/z,dot(d,uUp)*uFocal/z,0,1);}`,
    `#version 300 es\nprecision highp float;out vec4 c;void main(){c=vec4(.5,.9,.8,.8);}`);lineBuffer=gl.createBuffer();
}
function camera(){
  const im=meta.frames[frame],c=im.c2w;
  if(follow){eye=c.slice(0,3).map(r=>r[3]);right=c.slice(0,3).map(r=>r[0]);up=c.slice(0,3).map(r=>-r[1]);forward=c.slice(0,3).map(r=>r[2]);focal=im.K[1][1]/im.height*2;}
  else{const target=meta.view_target||[0,0,0];eye=[distance*Math.cos(pitch)*Math.sin(yaw),distance*Math.sin(pitch),distance*Math.cos(pitch)*Math.cos(yaw)].map((x,i)=>x+target[i]);forward=norm(target.map((x,i)=>x-eye[i]));right=norm(cross(forward,[0,1,0]));up=cross(right,forward);focal=1.8;}
}
function uniforms(p){gl.uniform3fv(gl.getUniformLocation(p,"uEye"),eye);gl.uniform3fv(gl.getUniformLocation(p,"uRight"),right);gl.uniform3fv(gl.getUniformLocation(p,"uUp"),up);gl.uniform3fv(gl.getUniformLocation(p,"uForward"),forward);gl.uniform2f(gl.getUniformLocation(p,"uSize"),canvas.width,canvas.height);gl.uniform1f(gl.getUniformLocation(p,"uFocal"),focal);}
function draw(){
  if(!gl||!points||!meta)return;const ratio=Math.min(devicePixelRatio,1.5);canvas.width=canvas.clientWidth*ratio;canvas.height=canvas.clientHeight*ratio;camera();
  gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.045,.068,.09,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.disable(gl.DEPTH_TEST);
  const n=points.length/17,order=Array.from({length:n},(_,i)=>i);order.sort((a,b)=>dot([points[b*17],points[b*17+1],points[b*17+2]],forward)-dot([points[a*17],points[a*17+1],points[a*17+2]],forward));
  const buffer=new Float32Array(points.length);
  order.forEach((id,j)=>{const off=id*17,to=j*17;buffer.set(points.subarray(off,off+17),to);if(mode>=0&&loadedStage===6){const nid=Math.round(points[off+14+mode]);const color=meta.palettes[mode][nid]||[.15,.15,.15];buffer.set(color,to+10);}if(selection){if(selection[id])buffer.set([.6,1,.78],to+10);else{buffer[to+10]*=.18;buffer[to+11]*=.18;buffer[to+12]*=.18;buffer[to+13]*=.25;}}});
  gl.useProgram(prog);gl.bindVertexArray(vao);uniforms(prog);gl.bindBuffer(gl.ARRAY_BUFFER,instanceBuffer);gl.bufferData(gl.ARRAY_BUFFER,buffer,gl.DYNAMIC_DRAW);gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,n);
  gl.bindVertexArray(null);gl.useProgram(lineProg);uniforms(lineProg);const path=[];meta.frames.slice(0,meta.stages[loadedStage-1].arrived).forEach(f=>path.push(...f.c2w.slice(0,3).map(r=>r[3])));gl.bindBuffer(gl.ARRAY_BUFFER,lineBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(path),gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);gl.drawArrays(gl.LINE_STRIP,0,path.length/3);gl.disableVertexAttribArray(0);
}
async function update(){
  if(!meta)return;const im=meta.frames[frame],s=meta.stages[im.stage-1];$("rgb").src=`/assets/input-${String(frame).padStart(3,"0")}.jpg`;$("render").src=`/assets/render-${String(frame).padStart(3,"0")}.jpg`;
  $("frame").textContent=`照片 ${frame+1} / ${meta.photographs}${im.test?" · 保留视角":""}`;$("stage").textContent=`批次 ${im.stage} · 已处理 ${s.arrived} 张`;
  $("count").textContent=s.gaussians.toLocaleString();$("quality").textContent=`${s.psnr.toFixed(2)} dB / ${s.ssim.toFixed(3)}`;$("time").textContent=`${s.chunk_seconds.toFixed(1)} s`;
  $("stepTime").textContent=`${s.steps} 步优化 · p50 ${s.update_p50_ms.toFixed(1)} ms / 步`;
  $("memory").textContent=`${s.gpu_peak_mib.toFixed(0)} MiB`;$("progress").textContent=`${frame+1} / ${meta.photographs}`;$("timeline").value=frame;
  $("modes").querySelectorAll("button").forEach(b=>b.disabled=im.stage!==6&&Number(b.dataset.mode)>=0);$("query").disabled=im.stage!==6;$("scale").disabled=im.stage!==6;$("threshold").disabled=im.stage!==6;
  if(im.stage!==6){queryRequest++;mode=-1;selection=null;seed=null;$("selection").textContent="历史批次显示几何；语义是完整地图的后处理结果。";}
  $("modes").querySelectorAll("button").forEach(b=>b.classList.toggle("active",Number(b.dataset.mode)===mode));
  if(loadedStage!==im.stage){const token=++request;const response=await fetch(`/assets/map-${im.stage}.bin`);const a=new Float32Array(await response.arrayBuffer());if(token!==request)return;points=a;loadedStage=im.stage;selection=null;seed=null;}
  draw();nodeList();$("status").textContent=`实际优化 ${s.gaussians.toLocaleString()} 点 · 当前显示 ${(points.length/17).toLocaleString()} 点 LOD · 批次结束快照`;
}
function nodeList(){const list=$("nodes");list.replaceChildren();if(loadedStage!==6||mode<0)return;meta.nodes.filter(n=>n.level===mode&&n.gaussians>50).sort((a,b)=>b.observations-a.observations).slice(0,45).forEach(n=>{const b=document.createElement("button");b.textContent=`${n.name} #${n.id} · ${n.observations} 视角`;b.onclick=()=>{selection=Array.from({length:points.length/17},(_,i)=>Math.round(points[i*17+14+mode])===n.id?1:0);$("selection").textContent=`节点 #${n.id} · ${n.gaussians.toLocaleString()} 个 Gaussian${n.parent?` · 父节点 #${n.parent}`:" · 父级未确定"}`;draw();};list.append(b);});}
async function query(path,payload){const token=++queryRequest;$("status").textContent="正在查询所学语义特征…";const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const a=await r.json();if(token!==queryRequest)return;if(!r.ok)throw Error(a.error);selection=a.selected;$("selection").textContent=`完整地图选中 ${a.full_map_selected.toLocaleString()} 个 Gaussian${a.node?` · 节点 #${a.node}`:""}`;draw();$("status").textContent="查询完成 · 本地 CPU 评估 Colab 训练的亲和特征";}
async function pointQuery(){if(seed===null||loadedStage!==6)return;try{await query('/api/point',{index:seed,scale:Math.exp(Number($("scale").value)),threshold:Number($("threshold").value)});}catch(e){$("status").textContent=e.message;}}
$("timeline").oninput=e=>{frame=Number(e.target.value);update();};$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Ⅱ 暂停":"▶ 播放照片序列";clearInterval(timer);if(playing)timer=setInterval(()=>{frame=(frame+1)%meta.photographs;update();},180);};
$("final").onclick=()=>{playing=false;clearInterval(timer);$("play").textContent="▶ 播放照片序列";frame=meta.photographs-1;update();};
$("modes").onclick=e=>{const b=e.target.closest('button');if(!b||b.disabled)return;queryRequest++;mode=Number(b.dataset.mode);selection=null;$("modes").querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));nodeList();draw();};
$("home").onclick=()=>{follow=true;draw();};$("clear").onclick=()=>{queryRequest++;selection=null;seed=null;$("selection").textContent="已清除选择；可点击地图上的 Gaussian。";draw();};
let debounce;$("scale").oninput=()=>{const scale=Math.exp(Number($("scale").value));$("scaleValue").textContent=scale.toFixed(3);clearTimeout(debounce);debounce=setTimeout(pointQuery,200);};
$("threshold").oninput=()=>{$("thresholdValue").textContent=Number($("threshold").value).toFixed(2);clearTimeout(debounce);debounce=setTimeout(pointQuery,200);};
$("query").onclick=async()=>{try{await query('/api/text',{word:$("word").value,level:Math.max(0,mode)});}catch(e){$("status").textContent=e.message;}};
let pointer=null,moved=0;canvas.onpointerdown=e=>{pointer=[e.clientX,e.clientY];moved=0;canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(!pointer)return;const dx=e.clientX-pointer[0],dy=e.clientY-pointer[1];moved+=Math.abs(dx)+Math.abs(dy);if(moved>4){follow=false;yaw-=dx*.005;pitch=Math.max(-1.4,Math.min(1.4,pitch+dy*.005));draw();}pointer=[e.clientX,e.clientY];};
canvas.onpointerup=e=>{pointer=null;if(moved>4||loadedStage!==6)return;const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=-1,d=144;camera();for(let i=0;i<points.length/17;i++){if(points[i*17+13]<.25)continue;const p=[0,1,2].map(j=>points[i*17+j]-eye[j]),z=dot(p,forward);if(z<=.01)continue;const px=rect.width*.5+dot(p,right)*focal*rect.height*.5/z,py=rect.height*.5-dot(p,up)*focal*rect.height*.5/z;const dist=(px-x)**2+(py-y)**2;if(dist<d){d=dist;best=i;}}if(best>=0){seed=best;pointQuery();}};
canvas.addEventListener('wheel',e=>{e.preventDefault();follow=false;distance=Math.max(.2,Math.min(6,distance*Math.exp(e.deltaY*.001)));draw();},{passive:false});
new ResizeObserver(()=>draw()).observe(canvas);
async function init(){
  meta=await(await fetch('/viewer.json')).json();$("timeline").max=meta.photographs-1;
  const scales=meta.semantic.query_scales;$("scale").min=Math.log(scales[2]*.6);$("scale").max=Math.log(scales[0]*1.5);$("scale").value=Math.log(scales[1]);$("scaleValue").textContent=scales[1].toFixed(3);
  const vocab=await(await fetch('/semantic-field/vocabulary.json')).json();vocab.forEach(word=>{const o=document.createElement('option');o.value=word;o.textContent=word;$("word").append(o);});
  const comparisons=await(await fetch('/comparison.json')).json();const table=document.createElement('table');table.innerHTML='<thead><tr><th>配置</th><th>Gaussian</th><th>PSNR</th><th>SSIM</th><th>旧视角变化</th></tr></thead>';
  const names={'kitchen-300k-l1':'30 万上限 + 回放','kitchen-600k-l1':'60 万上限 + 回放','kitchen-600k-no-replay':'60 万上限 · 关闭回放'};
  comparisons.forEach(r=>{const tr=document.createElement('tr');[names[r.variant],r.gaussians.toLocaleString(),r.psnr.toFixed(2),r.ssim.toFixed(3),`${r.old_change_db>=0?'+':''}${r.old_change_db.toFixed(2)} dB`].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td);});table.append(tr);});$("comparisons").append(table);await update();
}
init().catch(e=>$("status").textContent=e.message);
