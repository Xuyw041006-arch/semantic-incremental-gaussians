"""Desktop playback of actual CUDA map snapshots, without rerunning mapping."""
import argparse
import gzip
import json
from pathlib import Path
import threading
import numpy as np
from .data import ManifestSequence
from .mapping import MapConfig
from .server import serve,png,instance_color


class ReplaySession:
    def __init__(self,manifest,result):
        self.sequence=ManifestSequence(manifest); self.results=Path(result).resolve()
        translations={'remote':'遥控器','cup':'杯子','bottle':'瓶子','cell phone':'手机',
            'wine glass':'酒杯','bird':'鸟','cat':'猫','teddy bear':'玩具熊','chair':'椅子','tv':'显示器 / 电视','laptop':'笔记本电脑','mouse':'鼠标',
            'keyboard':'键盘','book':'书本','cardboard':'纸板','door-stuff':'门','floor-wood':'木地板',
            'wall-tile':'瓷砖墙','wall-wood':'木墙','window-other':'窗','cabinet-merged':'柜体',
            'table-merged':'桌面','floor-other-merged':'其他地面','paper-merged':'纸张','wall-other-merged':'其他墙面'}
        for label in self.sequence.labels: label['zh']=translations.get(label['name'],label.get('zh',label['name']))
        self.metadata=json.loads((self.results/'metadata.json').read_text())
        self.summary=json.loads((self.results/'summary.json').read_text())
        self.metrics=json.loads((self.results/'metrics.json').read_text())
        self.files=sorted((self.results/'replay').glob('*.json.gz'))
        if not self.files: raise ValueError('No actual map snapshots found in results/replay')
        self.lock=threading.Lock(); self.cursor=0; self.current=None
        with gzip.open(self.files[-1],'rt') as stream: last=json.load(stream)
        xyz=np.asarray(last['map']['xyz']).reshape(-1,3)
        center=np.median(xyz,axis=0); radius=np.quantile(np.linalg.norm(xyz-center,axis=1),.95)
        self.hint={'target':center.tolist(),'distance':float(np.clip(radius*2.6,2,12))}
        self.visible_labels=sorted(set(last['map']['semantic']))

    def state(self):
        empty={k:[] for k in ('ids','xyz','rgb','sigma','semantic','opacity','instance','confidence')}
        current=self.current or {}
        n=current.get('next_frame',0)
        return {'name':self.sequence.name,'description':'Colab T4 真实实验回放 · TUM desk · GT 相机位姿 · 每两帧保存优化后地图',
            'labels':self.sequence.labels,'visible_labels':self.visible_labels,'length':self.summary['train_frames'],
            'next_frame':n,'config':self.metadata['config'],'map':current.get('map',empty),
            'history':self.metrics[:n],'trajectory':current.get('trajectory',[]),'objects':current.get('objects',[]),
            'images':current.get('images'),'camera':current.get('camera'),'full':True,'replay':True,
            'backend':'Colab T4 · CUDA gsplat','cuda':True,'view_hint':self.hint}

    def step(self):
        if self.cursor>=len(self.files): return {'done':True}
        with gzip.open(self.files[self.cursor],'rt') as stream: current=json.load(stream)
        self.cursor+=1
        frame=self.sequence.get(current['source_frame'])
        table=np.asarray([l['color'] for l in self.sequence.labels],np.uint8)
        current['images']={'rgb':png(frame.rgb),'semantic':png(table[frame.semantic]),'instance':png(instance_color(frame.instance))}
        current['stats']['cached_segmentation_ms']=self.sequence.frames[current['source_frame']].get('segmentation_ms')
        current['history']=self.metrics[:current['next_frame']]
        current['history'][-1]=current['stats']
        current['full']=True; current['done']=self.cursor>=len(self.files)
        self.current=current
        return current

    def reset(self,payload):
        self.cursor=0; self.current=None
        return self.state()

    def export(self):
        return {'path':str(self.results/'map.ply'),'gaussians':self.summary['gaussians']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True); parser.add_argument('--result',required=True)
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args(); session=ReplaySession(args.manifest,args.result)
    serve(session.sequence,MapConfig(**session.metadata['config']),args.result,args.port,session=session)


if __name__=='__main__': main()
