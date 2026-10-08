"""Run this first on Colab to verify the installed gsplat forward/backward contract."""
import json


def main():
    import torch
    from gsplat import rasterization
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU runtime is required")
    means=torch.tensor([[0.,0.,2.]],device="cuda",requires_grad=True)
    colors=torch.tensor([[.8,.2,.1]],device="cuda",requires_grad=True)
    quats=torch.tensor([[1.,0.,0.,0.]],device="cuda")
    scales=torch.full((1,3),.12,device="cuda",requires_grad=True)
    opacity=torch.tensor([.8],device="cuda",requires_grad=True)
    K=torch.tensor([[[50.,0.,31.5],[0.,50.,31.5],[0.,0.,1.]]],device="cuda")
    view=torch.eye(4,device="cuda")[None]
    rgb,alpha,_=rasterization(means,quats,scales,opacity,colors,view,K,64,64,render_mode="RGB+ED",packed=True)
    assert rgb.shape==(1,64,64,4) and alpha.shape==(1,64,64,1)
    loss=rgb[...,:3].square().mean()+.01*rgb[...,3].mean(); loss.backward()
    for name,value in [("means",means),("colors",colors),("scales",scales),("opacity",opacity)]:
        assert value.grad is not None and torch.isfinite(value.grad).all(),name
    assert colors.grad.abs().sum()>0
    # The semantic feature renderer uses an arbitrary channel count, including >128.
    features=torch.softmax(torch.randn(1,134,device="cuda"),dim=-1).requires_grad_()
    sem,_,_=rasterization(means.detach(),quats,scales.detach(),opacity.detach(),features,view,K,64,64,packed=True)
    sem.mean().backward(); assert features.grad is not None and torch.isfinite(features.grad).all()
    print(json.dumps({"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
                      "rgb_depth_shape":list(rgb.shape),"semantic_shape":list(sem.shape),"backward":"passed"},indent=2))


if __name__=="__main__": main()
