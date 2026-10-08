"""Pure NumPy hierarchy and robust multi-view descriptor utilities.

Containment is evidence of nesting, not proof that a mask denotes a named part.
Unknown/uncovered pixels and conflicting hierarchy assignments remain explicit.
"""
import numpy as np


def containment_tree(masks, threshold=.9):
    masks=np.asarray(masks,dtype=bool); n=len(masks)
    flat=masks.reshape(n,-1); areas=flat.sum(1)
    parents=np.full(n,-1,np.int32)
    for i in range(n):
        if not areas[i]: continue
        larger=np.where(areas>areas[i]*1.1)[0]
        larger=larger[np.argsort(areas[larger])]
        for j in larger:
            if np.count_nonzero(flat[i]&flat[j])/areas[i]>=threshold:
                parents[i]=j; break
    return parents


def scale_layers(masks, scales, targets=(.7,.25,.08)):
    """Smallest containing mask above a query scale, in normalized scene units."""
    masks=np.asarray(masks,dtype=bool); scales=np.asarray(scales)
    layers=np.zeros((len(targets),*masks.shape[1:]),np.int32)
    for level,target in enumerate(targets):
        # Overwrite large masks with smaller masks; zero means no supported mask.
        for i in np.argsort(scales)[::-1]:
            if scales[i]>=target: layers[level][masks[i]]=int(i)+1
    return layers


def mask_physical_scale(mask,depth,K,c2w=None):
    y,x=np.where(mask&(depth>0)&np.isfinite(depth))
    if len(x)<20: return 0.
    take=np.linspace(0,len(x)-1,min(4000,len(x))).astype(int); x=x[take]; y=y[take]
    z=depth[y,x]
    points=np.column_stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z])
    # Rotation invariant robust extent; discard extreme background depth outliers.
    lo,hi=np.percentile(z,[5,95]); points=points[(z>=lo)&(z<=hi)]
    return float(2*np.sqrt(np.var(points,axis=0).sum()))


def multiview_descriptors(features,max_clusters=4):
    """LaGa-inspired adaptive prototypes, not a reproduction of original LaGa.

    Farthest-first splitting with cosine separation followed by weighted K-means.
    Weights reflect alignment with global semantics and cluster compactness.
    """
    x=np.asarray(features,dtype=np.float32); x=x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-8)
    global_feature=x.mean(0); global_feature/=max(np.linalg.norm(global_feature),1e-8)
    centers=[x[np.argmax(x@global_feature)]]
    for _ in range(1,min(max_clusters,len(x))):
        sim=x@np.asarray(centers).T; j=np.argmin(sim.max(1))
        if sim[j].max()>.88: break
        centers.append(x[j])
    centers=np.asarray(centers)
    for _ in range(12):
        labels=np.argmax(x@centers.T,axis=1)
        updated=[]
        for j in range(len(centers)):
            v=x[labels==j].mean(0) if np.any(labels==j) else centers[j]
            updated.append(v/max(np.linalg.norm(v),1e-8))
        centers=np.asarray(updated)
    weights=[]
    for j,c in enumerate(centers):
        assigned=x[labels==j]
        compact=float(np.mean(assigned@c)) if len(assigned) else 0
        weights.append(max(0,float(c@global_feature))*max(0,compact)*len(assigned))
    weights=np.asarray(weights); weights/=max(weights.sum(),1e-8)
    return centers,weights


def infer_parent_ids(coarse,fine,min_support=.6):
    """Return child→parent only when majority support is sufficient."""
    result={}
    for child in np.unique(fine):
        if child<=0: continue
        values,counts=np.unique(coarse[fine==child],return_counts=True)
        valid=values>0
        if not valid.any(): result[int(child)]=0; continue
        ids=values[valid]; nums=counts[valid]; j=np.argmax(nums)
        result[int(child)]=int(ids[j]) if nums[j]/np.count_nonzero(fine==child)>=min_support else 0
    return result
