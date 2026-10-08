"""Local-only viewer. Playback requests one real integration step at a time."""
import base64
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import numpy as np
from PIL import Image
from .mapping import GaussianMap, MapConfig

WEB=Path(__file__).resolve().parent/"web"


def png(array):
    b=BytesIO(); Image.fromarray(array).save(b,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()


def instance_color(ids):
    ids=np.asarray(ids,dtype=np.int64)
    result=np.stack([70+(ids*53)%170,70+(ids*97)%170,70+(ids*131)%170],-1).astype(np.uint8)
    result[ids==0]=[48,56,67]
    return result


class Session:
    def __init__(self, sequence, config, results):
        self.sequence=sequence; self.mapping=GaussianMap(sequence.labels,config)
        self.results=Path(results); self.lock=threading.Lock(); self.last_images=None

    def state(self):
        return {"name":self.sequence.name,"description":self.sequence.description,"labels":self.sequence.labels,
                "length":len(self.sequence),"next_frame":self.mapping.frame_id,"config":asdict(self.mapping.config),
                "map":self.mapping.packet(),"history":self.mapping.history,"trajectory":self.mapping.trajectory,
                "objects":self.mapping.tracker.summary(),"images":self.last_images,"full":True,
                "backend":"CPU · isotropic Gaussian fusion", "cuda":False}

    def step(self):
        index=self.mapping.frame_id
        if index>=len(self.sequence): return {"done":True}
        begin=time.perf_counter(); frame=self.sequence.get(index); source_ms=(time.perf_counter()-begin)*1000
        stats,changed,full=self.mapping.integrate(frame)
        stats["source_ms"]=round(source_ms,3)
        if hasattr(self.sequence,"frames"):
            stats["cached_segmentation_ms"]=self.sequence.frames[index].get("segmentation_ms")
        table=np.array([l["color"] for l in self.sequence.labels],np.uint8)
        self.last_images={"rgb":png(frame.rgb),"semantic":png(table[frame.semantic]),"instance":png(instance_color(frame.instance))}
        packet=self.mapping.packet(None if full else changed)
        return {"done":index+1>=len(self.sequence),"stats":stats,"map":packet,"full":full,
                "trajectory":self.mapping.trajectory,"objects":self.mapping.tracker.summary(),"images":self.last_images,
                "camera":frame.c2w.round(5).tolist(),"next_frame":self.mapping.frame_id}

    def reset(self,payload):
        old=asdict(self.mapping.config)
        for key in ("max_gaussians","target_update_ms"):
            if key in payload: old[key]=float(payload[key])
        old["max_gaussians"]=int(np.clip(old["max_gaussians"],100,100000))
        old["target_update_ms"]=float(np.clip(old["target_update_ms"],5,1000))
        self.mapping=GaussianMap(self.sequence.labels,MapConfig(**old)); self.last_images=None
        return self.state()

    def export(self):
        directory=self.results/datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
        self.mapping.export(directory)
        return {"path":str(directory.resolve()),"gaussians":self.mapping.count}


def serve(sequence,config,results,port=8765,session=None):
    session=session or Session(sequence,config,results)
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,body,content="application/json"):
            if content=="application/json": body=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(code); self.send_header("Content-Type",content)
            self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=="/api/state":
                with session.lock: self.send(200,session.state())
            elif path in ("/","/app.js","/style.css"):
                name={"/":"index.html","/app.js":"app.js","/style.css":"style.css"}[path]
                mime={"index.html":"text/html; charset=utf-8","app.js":"application/javascript; charset=utf-8","style.css":"text/css; charset=utf-8"}[name]
                self.send(200,(WEB/name).read_bytes(),mime)
            else: self.send(404,{"error":"Not found"})

        def do_POST(self):
            # Only JSON requests from this local viewer may mutate the session.
            origin=self.headers.get("Origin")
            if origin and urlsplit(origin).netloc!=self.headers.get("Host"):
                self.send(403,{"error":"Origin mismatch"}); return
            if "application/json" not in self.headers.get("Content-Type",""):
                self.send(415,{"error":"JSON required"}); return
            try:
                length=int(self.headers.get("Content-Length",0))
                if not 0<=length<=4096: raise ValueError("request too large")
                payload=json.loads(self.rfile.read(length) or b"{}")
                with session.lock:
                    if self.path=="/api/step": self.send(200,session.step())
                    elif self.path=="/api/reset":
                        self.send(200,session.reset(payload))
                    elif self.path=="/api/export":
                        self.send(200,session.export())
                    else: self.send(404,{"error":"Not found"})
            except (ValueError,TypeError,KeyError) as e: self.send(400,{"error":str(e)})
            except Exception as e:
                import traceback
                traceback.print_exc(); self.send(500,{"error":str(e)})

        def log_message(self,fmt,*args):
            if args and str(args[1]) not in ("200","304"): super().log_message(fmt,*args)
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"Viewer: http://127.0.0.1:{port}  |  {sequence.description}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
