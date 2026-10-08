"""Assemble actual held-out renders, without fabricating or regenerating images."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path.cwd()

def main():
    for ds in ('tum-desk','kitchen','tum-desk2'):
        base=ROOT/'results'/ds/'000'/'seed-7';full=ROOT/'results'/ds/'111'/'seed-7'
        candidates=sorted(base.glob('semantic-6-*.jpg'))
        assert candidates
        for which,path in [('early',candidates[0]),('late',candidates[-1])]:
            index=int(path.stem.split('-')[-1]);fig,axes=plt.subplots(2,4,figsize=(13,5.8),constrained_layout=True)
            for row,(folder,label) in enumerate([(base,'Baseline 000'),(full,'Full 111')]):
                semantic=np.asarray(Image.open(folder/path.name));h,w=semantic.shape[:2];tile=w//3
                view=Image.open(folder/f'view-6-{index:04d}.jpg');vw,vh=view.size
                pred=np.asarray(view.crop((vw//2,0,vw,vh)).resize((tile,h),Image.Resampling.LANCZOS))
                images=[semantic[:,:tile],pred,semantic[:,tile:tile*2],semantic[:,tile*2:]]
                for col,(im,title) in enumerate(zip(images,['Reference RGB','Reconstructed RGB','Object candidate regions','Fine candidate regions'])):
                    axes[row,col].imshow(im);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                    if row==0:axes[row,col].set_title(title,fontsize=10)
                    if col==0:axes[row,col].set_ylabel(label,fontsize=11)
            fig.suptitle(f'{ds} | held-out frame {index} ({which}) | final stage, seed 7\nRegion colors encode persistent IDs; candidates are SAM-derived, not human part ground truth.',fontsize=11)
            fig.savefig(ROOT/'analysis'/f'{ds}-visual-{which}.png',dpi=150);plt.close(fig)
    print('ACTUAL_RENDER_COMPARISON_PANELS_SAVED')

if __name__=='__main__':main()
