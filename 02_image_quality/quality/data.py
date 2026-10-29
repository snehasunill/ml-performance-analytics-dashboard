"""Build grouped, reproducible four-class image datasets from user-owned photos."""
import argparse, csv, hashlib, io, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

LABELS = ('clean','blur','noise','jpeg')
EXTS={'.png','.jpeg','.jpg','.webp','.bmp'}

def corrupt(img, label, seed, severity=0.65):
    if label not in LABELS: raise ValueError('unknown corruption')
    if not 0 <= severity <= 1: raise ValueError('severity out of range')
    img=img.convert('RGB')
    if label=='clean': return img.copy()
    if label=='blur': return img.filter(ImageFilter.GaussianBlur(radius=0.8+severity*4.2))
    if label=='noise':
        ar=np.asarray(img).astype(np.float32)
        ar=np.clip(ar+np.random.default_rng(seed).normal(0,8+severity*45, ar.shape),0,255).astype('uint8')
        return Image.fromarray(ar)
    b=io.BytesIO(); img.save(b,format='JPEG',quality=max(2,round(50-46*severity)));b.seek(0)
    return Image.open(b).convert('RGB').copy()

def source_split(source_key, seed=42, train=.7, val=.15):
    if not (0<train<1 and 0<val<1 and train+val<1): raise ValueError('bad split')
    n=int(hashlib.sha256(f'{seed}/{source_key}'.encode()).hexdigest()[:12],16)/float(16**12)
    return 'train' if n<train else ('val' if n<train+val else 'test')

def image_files(root):
    return sorted(x for x in Path(root).rglob('*') if x.is_file() and x.suffix.lower() in EXTS)

def build(source_dir, output_dir, variants_per_source=10, seed=42, size=224):
    """Variants include clean images; deterministic hash split occurs BEFORE augmentations."""
    if variants_per_source<4 or size<32: raise ValueError('need >=4 variants and size>=32')
    src=Path(source_dir); out=Path(output_dir)
    if out.resolve()==src.resolve() or src.resolve() in out.resolve().parents: raise ValueError('output inside source')
    sources=image_files(src)
    if not sources: raise ValueError('no image sources found')
    out.mkdir(parents=True,exist_ok=True)
    manifest=[]; discarded=[]
    for index, fp in enumerate(sources):
        key=str(fp.relative_to(src))
        try:
            with Image.open(fp) as original:
                image=ImageOps.exif_transpose(original).convert('RGB')
                if min(image.size)<32: raise ValueError('image too small')
                image=ImageOps.fit(image,(size,size),method=Image.Resampling.LANCZOS)
        except (OSError,ValueError) as e:
            discarded.append({'source':key,'reason':str(e)});continue
        split=source_split(key,seed)
        for j in range(variants_per_source):
            label=LABELS[j%4]; local_seed=seed+index*100003+j
            severity=.35+(j%5)*.14
            im=corrupt(image,label,local_seed,severity)
            name=f'{index:06}_{j:03}.png'; path=out/split/label/name
            path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
            manifest.append({'path':str(path.relative_to(out)),'source_id':key,'split':split,'label':label,'severity':severity})
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','source_id','split','label','severity']);w.writeheader();w.writerows(manifest)
    (out/'build_report.json').write_text(json.dumps({'input_sources':len(sources),'valid_sources':len(sources)-len(discarded),'variants':len(manifest),'excluded':discarded,'synthetic':True},indent=2))
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--variants',type=int,default=10);ap.add_argument('--size',type=int,default=224)
    args=ap.parse_args();print('Generated',len(build(args.sources,args.output,args.variants,size=args.size)),'variants')
