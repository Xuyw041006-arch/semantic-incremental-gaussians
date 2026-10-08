"""Optional learned panoptic segmentation. Weights are fetched only on explicit run."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

MODEL = "facebook/mask2former-swin-tiny-coco-panoptic"


def decode_panoptic(segmentation, segments_info, shape, thing_labels):
    semantic=np.zeros(shape,np.int32); instance=np.zeros(shape,np.int32); confidence=np.zeros(shape,np.float32)
    if segmentation is None:
        return semantic,instance,confidence
    segmentation=np.asarray(segmentation)
    for info in segments_info:
        mask=segmentation==info["id"]; label=int(info["label_id"])
        semantic[mask]=label+1  # reserve zero for unknown, including unassigned -1 pixels
        if label in thing_labels: instance[mask]=int(info["id"])+1
        confidence[mask]=float(info["score"])
    return semantic,instance,confidence


def segment_manifest(manifest_path, output_path, device="auto", model_name=MODEL):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA not available")
    source=Path(manifest_path).resolve(); output=Path(output_path).resolve()
    # Keep RGB/depth paths valid, and never overwrite the source manifest.
    if output==source or output.parent!=source.parent:
        raise ValueError("output must be a new manifest in the same dataset directory")
    m=json.loads(source.read_text()); root=source.parent
    processor=AutoImageProcessor.from_pretrained(model_name)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device).eval()
    id2label={int(k):v for k,v in model.config.id2label.items()}
    if sorted(id2label)!=list(range(len(id2label))): raise ValueError("model labels must be contiguous")
    # COCO panoptic train IDs 0..79 are things. Other models require explicit metadata.
    if model_name!=MODEL: raise ValueError("custom model requires an explicit thing/stuff label mapping")
    things=set(range(80)); stuff=set(id2label)-things
    labels=[{"name":"unknown","zh":"未知","thing":False,"color":[113,123,137]}]
    for i in range(len(id2label)):
        labels.append({"name":id2label[i],"zh":id2label[i],"thing":i in things,
                       "color":[int(70+(i*53)%160),int(70+(i*97)%160),int(70+(i*131)%160)]})
    folder=root/(output.stem+"_masks"); folder.mkdir(exist_ok=True)
    timings=[]
    for index,f in enumerate(m["frames"]):
        image_path=(root/f["rgb"]).resolve()
        if not image_path.is_relative_to(root): raise ValueError("image outside dataset")
        image=Image.open(image_path).convert("RGB")
        inputs=processor(images=image,return_tensors="pt").to(device)
        if device=="cuda": torch.cuda.synchronize()
        begin=time.perf_counter()
        with torch.inference_mode(): outputs=model(**inputs)
        pred=processor.post_process_panoptic_segmentation(outputs,target_sizes=[(image.height,image.width)],
              label_ids_to_fuse=stuff)[0]
        if device=="cuda": torch.cuda.synchronize()
        timings.append((time.perf_counter()-begin)*1000)
        seg=pred["segmentation"]
        if seg is not None: seg=seg.cpu().numpy()
        semantic,instance,confidence=decode_panoptic(seg,pred["segments_info"],(image.height,image.width),things)
        for name,array in [("semantic",semantic),("instance",instance),("confidence",confidence)]:
            path=folder/f"{index:06}_{name}.npy"; np.save(path,array)
            f[name]=str(path.relative_to(root))
        f["segmentation_source"]="mask2former_prediction"
        f["segmentation_ms"]=round(timings[-1],3)
        print(f"segment {index+1}/{len(m['frames'])}: {timings[-1]:.1f} ms",flush=True)
    m["labels"]=labels; m["segmentation_model"]=model_name
    m["segmentation_device"]=device
    m["segmentation_note"]="Cached predictions; segmentation time is separate from online mapping time."
    output.write_text(json.dumps(m,ensure_ascii=False,indent=2))
    return output
