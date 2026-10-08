"""Run only after the mapper exits: isolated teacher latency and memory audit."""
from pathlib import Path
import json,time,threading,subprocess
import numpy as np
from PIL import Image
import torch
import sys
sys.path.insert(0,str(Path.cwd()))
from hglab.readers import load_scene
from hglab.colmap import image_rgb

ROOT=Path.cwd()

def main():
    assert json.loads((ROOT/'status.json').read_text())['state']=='complete'
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import open_clip
    torch.set_num_threads(2);torch.manual_seed(7)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:samples.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
            except Exception:pass
            stop.wait(.2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();load=time.perf_counter()
    checkpoint=list(Path('/content/hierarchical-gaussian').rglob('sam2.1_hiera_tiny.pt'))[0]
    sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(checkpoint),device='cuda',apply_postprocessing=False)
    generator=SAM2AutomaticMaskGenerator(sam,points_per_side=16,points_per_batch=64,pred_iou_thresh=.75,
        stability_score_thresh=.9,crop_n_layers=1,crop_n_points_downscale_factor=2,box_nms_thresh=.7)
    clip,_,prep=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True);clip.eval()
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load;torch.cuda.reset_peak_memory_stats();rows=[]
    plan=json.loads((ROOT/'experiment-plan.json').read_text())
    for ds in plan['datasets']:
        scene=load_scene(ds['data'],640)
        for im in [im for im in scene['images'] if not im['test']][::6][:3]:
            start=time.perf_counter();rgb=image_rgb(im)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):annotations=generator.generate(rgb)
            annotations=sorted([a for a in annotations if a['area']>=max(100,rgb.shape[0]*rgb.shape[1]*.0003)],
                key=lambda a:a['predicted_iou']*a['stability_score'],reverse=True)[:100]
            crops=[]
            for a in annotations:
                x,y,w,h=map(int,a['bbox']);masked=rgb.copy();masked[~a['segmentation']]=127
                crops.append(prep(Image.fromarray(masked[y:y+h,x:x+w])))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                for j in range(0,len(crops),16):
                    features=clip.encode_image(torch.stack(crops[j:j+16]).cuda()).float()
                    features=features/features.norm(dim=-1,keepdim=True)
                    features.cpu().numpy()
            torch.cuda.synchronize();rows.append(dict(dataset=ds['name'],frame=im['index'],masks=len(crops),
                seconds=time.perf_counter()-start,gpu_allocated_peak_mib=torch.cuda.max_memory_allocated()/1024**2))
    stop.set();thread.join()
    result=dict(device=torch.cuda.get_device_name(),model_load_seconds=load_seconds,frames=rows,
        whole_gpu_sampled_peak_mib=max(samples),whole_gpu_sampling_interval_seconds=.2,
        qualification='Separate isolated SAM+CLIP benchmark after mapping; includes RGB load and crops; excludes pose estimation. First frame includes warm-up.')
    (ROOT/'teacher-benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
