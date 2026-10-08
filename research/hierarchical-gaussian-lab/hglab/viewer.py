"""Local read-only replay and actual learned-affinity scale query."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
import argparse,json
import numpy as np


def scale_gate(weights,scale):
    x=np.array([np.log(max(.01,float(scale)))],np.float32)
    hidden=np.maximum(0,x@weights['0.weight'].T+weights['0.bias'])
    logits=hidden@weights['2.weight'].T+weights['2.bias']
    return 1/(1+np.exp(-logits))


class QueryEngine:
    def __init__(self,root):
        self.root=Path(root); sem=self.root/'semantic-field'
        self.features=np.load(sem/'affinity-fp16.npy',mmap_mode='r')
        self.preview=np.load(self.root/'assets/preview-indices.npy')
        self.weights=dict(np.load(sem/'gate-weights.npz'))
        self.hierarchy=np.load(sem/'hierarchy.npz')
        self.nodes=json.loads((sem/'nodes.json').read_text())
        self.vocab=json.loads((sem/'vocabulary.json').read_text())
        self.text=np.load(sem/'text-features.npy')

    def point(self,index,scale,threshold=.75):
        index=int(index)
        if not 0<=index<len(self.preview): raise ValueError('Invalid displayed Gaussian index')
        g=scale_gate(self.weights,scale)
        seed=self.features[self.preview[index]].astype('float32')*g; seed/=max(np.linalg.norm(seed),1e-8)
        scores=[]; count=0
        for start in range(0,len(self.features),100000):
            f=self.features[start:start+100000].astype('float32')*g
            sim=(f@seed)/np.linalg.norm(f,axis=1).clip(1e-8)
            count+=int((sim>=threshold).sum()); scores.append(sim)
        scores=np.concatenate(scores)
        return dict(selected=(scores[self.preview]>=threshold).astype('uint8').tolist(),
                    full_map_selected=count,scale=float(scale),threshold=threshold)

    def text_query(self,word,level):
        if word not in self.vocab: raise ValueError('本地预设词表没有该查询；任意文本可在 Colab 的 hglab.query 中编码。')
        q=self.text[self.vocab.index(word)]
        proto=self.hierarchy['prototypes'].astype('float32'); weights=self.hierarchy['weights']
        ranking=[]
        for node in self.nodes:
            if node['level']!=int(level) or node['gaussians']<20: continue
            b=node['descriptor_begin']; e=b+node['descriptor_count']
            score=float(np.sum((proto[b:e]@q)*weights[b:e])); ranking.append((score,node['id']))
        ranking.sort(reverse=True); chosen=ranking[0][1] if ranking else 0
        ids=self.hierarchy['ids'][int(level)]
        return dict(selected=(ids[self.preview]==chosen).astype('uint8').tolist(),
                    full_map_selected=int((ids==chosen).sum()),node=chosen,
                    ranking=ranking[:5],word=word)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,root,engine,**kw): self.root=Path(root); self.engine=engine; super().__init__(*a,directory=str(root),**kw)
    def do_GET(self):
        if self.path=='/': self.path='/web/index.html'
        super().do_GET()
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n>4096: raise ValueError('Request too large')
            a=json.loads(self.rfile.read(n))
            if self.path=='/api/point': result=self.engine.point(a['index'],a['scale'],a.get('threshold',.75))
            elif self.path=='/api/text': result=self.engine.text_query(a['word'],a['level'])
            else: raise ValueError('Unknown query')
            data=json.dumps(result).encode(); self.send_response(200)
        except (ValueError,KeyError) as e:
            data=json.dumps(dict(error=str(e)),ensure_ascii=False).encode(); self.send_response(400)
        self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def log_message(self,*a): pass


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--port',type=int,default=8767); a=p.parse_args()
    root=Path(a.data).resolve(); web=Path(__file__).with_name('web')
    import shutil
    shutil.copytree(web,root/'web',dirs_exist_ok=True)
    engine=QueryEngine(root)
    print('Hierarchical real Gaussian replay: http://127.0.0.1:'+str(a.port),flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,root=root,engine=engine)).serve_forever()
