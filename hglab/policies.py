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


def sampling_weights(indices, coverage, births, stage, errors=None, priorities=None):
    """Bounded uniform mixture: RGB difficulty dominates mild semantic novelty."""
    errors=errors or {}
    scores=[]
    for i in indices:
        regions=coverage.get(int(i),[])
        novelty=sum(births.get(str(n),-1)==stage for n in regions)/max(1,len(regions))
        error=np.clip(float(errors.get(int(i),.05))/.05,.5,2.)
        task=max([float((priorities or {}).get(str(int(n)//2),{}).get('priority',1.)) for n in regions]+[1.])
        scores.append(error*(1+.25*novelty)*(1+.5*(task-1)))
    a=np.asarray(scores,float);a/=a.sum()
    return .5/len(indices)+.5*a


def replay_probability(stage,max_probability=.6):
    """Old observations gradually occupy more of the fixed optimization budget."""
    return .3+(max_probability-.3)*(stage-1)/max(1,stage)


def coverage_replay(indices, coverage, capacity=48, poses=None, priorities=None):
    """Greedy weighted set cover, with pose novelty and unknown-frame exploration."""
    indices=list(dict.fromkeys(map(int,indices)))
    if len(indices)<=capacity:return indices
    frequency={}
    for i in indices:
        for n in set(coverage.get(i,[])):frequency[n]=frequency.get(n,0)+1
    weights={n:(2. if n%2 else 1.)*(1+float((priorities or {}).get(str(int(n)//2),{}).get('priority',1.)))/np.sqrt(v) for n,v in frequency.items()}
    chosen=[];covered=set();remaining=set(indices)
    # Reserve 1/3 of the pool for uniformly spaced frames, including unknown ones.
    explore=min(max(1,capacity//3),len(indices))
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


def choose_crop(im,region_boxes,rng,semantic=False,priorities=None):
    """Identical crop dimensions across ablations; only locations differ."""
    w=min(384,im['width']);h=min(256,im['height'])
    if semantic and region_boxes and rng.random()<.4:
        weights=np.asarray([1+float((priorities or {}).get(str(r['node']),{}).get('priority',1.)) for r in region_boxes])
        region=region_boxes[int(rng.choice(len(region_boxes),p=weights/weights.sum()))]
        cx,cy=region['center'];cx+=rng.normal(0,w*.12);cy+=rng.normal(0,h*.12)
    else:cx=rng.uniform(w/2,im['width']-w/2);cy=rng.uniform(h/2,im['height']-h/2)
    x=int(np.clip(round(cx-w/2),0,im['width']-w));y=int(np.clip(round(cy-h/2),0,im['height']-h))
    return crop_camera(im,x,y,w,h),(x,y,w,h)
