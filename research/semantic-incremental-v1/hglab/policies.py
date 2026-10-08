"""Budget and replay policies, independent of CUDA and future observations."""
import numpy as np


def semantic_importance(ids, confidence, min_conf=.75):
    """Region-balanced marginal coverage proxy; uncertainty alone earns no bonus."""
    ids=np.asarray(ids); confidence=np.asarray(confidence)
    score=np.zeros(ids.shape[1],np.float32)
    for level,weight in enumerate((1.,2.)):
        known=(ids[level]>0)&(confidence[level]>=min_conf)
        counts=np.bincount(ids[level,known],minlength=int(ids[level].max(initial=0))+1)
        if known.any():
            # Cap amplification and normalize each level to a comparable scale.
            marginal=1/np.sqrt(counts[ids[level,known]].clip(1))
            marginal/=max(float(np.mean(marginal)),1e-8)
            score[known]+=weight*confidence[level,known]*np.minimum(marginal,4)
    return score


def sampling_weights(indices, coverage, births, stage, errors=None):
    scores=[]
    errors=errors or {}
    for i in indices:
        regions=coverage.get(int(i),[])
        novelty=sum(births.get(str(n),-1)==stage for n in regions)/max(1,len(regions))
        # Positive exploration probability protects unlabeled and unknown surfaces.
        scores.append(.25+novelty+min(2.,float(errors.get(int(i),.05))/.1))
    a=np.asarray(scores,float); return a/a.sum()


def coverage_replay(indices, coverage, capacity=48, poses=None):
    """Greedy weighted set cover, with pose novelty and unknown-frame exploration."""
    indices=list(dict.fromkeys(map(int,indices)))
    if len(indices)<=capacity:return indices
    frequency={}
    for i in indices:
        for n in set(coverage.get(i,[])):frequency[n]=frequency.get(n,0)+1
    weights={n:(2. if n%2 else 1.)/np.sqrt(v) for n,v in frequency.items()}
    chosen=[];covered=set();remaining=set(indices)
    # Reserve 1/8 of the pool for uniformly spaced frames, including unknown ones.
    explore=min(max(1,capacity//8),len(indices))
    for j in np.linspace(0,len(indices)-1,explore).round().astype(int):
        i=indices[j]
        if i in remaining:chosen.append(i);remaining.remove(i);covered.update(coverage.get(i,[]))
    while len(chosen)<capacity:
        best=None;best_score=-1.
        for i in sorted(remaining):
            gain=sum(weights[n] for n in set(coverage.get(i,[]))-covered)
            if poses is not None:
                dist=min(np.linalg.norm(poses[i]-poses[j]) for j in chosen)
                gain+=.05*min(float(dist),2.)
            # Deterministic tiny recency tie-break, never dependent on validation.
            gain+=1e-8*i
            if gain>best_score:best=i;best_score=gain
        chosen.append(best);remaining.remove(best);covered.update(coverage.get(best,[]))
    return chosen


def crop_camera(im,x,y,w,h):
    result=im.copy();result['K']=im['K'].copy()
    result['K'][0,2]-=x;result['K'][1,2]-=y
    result['width']=w;result['height']=h
    return result


def choose_crop(im,region_boxes,rng,semantic=False):
    """Identical crop dimensions across ablations; only locations differ."""
    w=min(384,im['width']);h=min(256,im['height'])
    if semantic and region_boxes and rng.random()<.8:
        region=region_boxes[int(rng.integers(len(region_boxes)))]
        cx,cy=region['center'];cx+=rng.normal(0,w*.12);cy+=rng.normal(0,h*.12)
    else:cx=rng.uniform(w/2,im['width']-w/2);cy=rng.uniform(h/2,im['height']-h/2)
    x=int(np.clip(round(cx-w/2),0,im['width']-w));y=int(np.clip(round(cy-h/2),0,im['height']-h))
    return crop_camera(im,x,y,w,h),(x,y,w,h)
