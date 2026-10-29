"""Train baselines and save reproducible validation/test metrics, never a fabricated score."""
import argparse,csv,json,random,time
from pathlib import Path
import joblib,numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .data import LABELS
from .models import classical,features,neural,transform

def read_manifest(root):
    with (Path(root)/'manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('empty manifest')
    groups={}
    for r in rows:
        if r['label'] not in LABELS or r['split'] not in {'train','val','test'}:raise ValueError('bad manifest')
        if r['source_id'] in groups and groups[r['source_id']]!=r['split']:raise ValueError('SOURCE LEAKAGE')
        groups[r['source_id']]=r['split']
    if not all(any(r['split']==s for r in rows) for s in ('train','val','test')):raise ValueError('need nonempty train/val/test; add more sources')
    return rows

def report(y,p):
    return {'macro_f1':float(f1_score(y,p,labels=list(LABELS),average='macro',zero_division=0)),
      'accuracy':float(accuracy_score(y,p)), 'confusion_matrix':confusion_matrix(y,p,labels=LABELS).tolist(),
      'per_class':classification_report(y,p,labels=LABELS,output_dict=True,zero_division=0)}

def split_rows(rows,s):return [r for r in rows if r['split']==s]
def classical_train(root,out,seed=42):
    rows=read_manifest(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def xy(s):
        rs=split_rows(rows,s)
        x=np.stack([features(Image.open(Path(root)/r['path'])) for r in rs]);y=[r['label'] for r in rs]
        return x,y,rs
    x,y,_=xy('train');model=classical(seed);model.fit(x,y)
    metrics={}
    for split in ('val','test'):
        xx,yy,rs=xy(split);pred=model.predict(xx)
        metrics[split]=report(yy,pred)
        with (out/f'{split}_predictions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','source_id','truth','prediction']);w.writeheader()
            w.writerows([{'path':r['path'],'source_id':r['source_id'],'truth':r['label'],'prediction':v} for r,v in zip(rs,pred)])
    joblib.dump(model,out/'model.joblib');(out/'metadata.json').write_text(json.dumps({'model':'opencv','labels':LABELS,'metrics':metrics,'n_train':len(y),'seed':seed,'size':128,'note':'size must match dataset generation resolution'},indent=2))
    return metrics

def torch_train(root,out,kind='cnn',epochs=3,batch=24,size=128,lr=.001,seed=42,pretrained=False):
    import torch
    from torch.utils.data import DataLoader,Dataset
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rows=read_manifest(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    t=transform(size,kind)
    class Photos(Dataset):
        def __init__(self,split):self.rows=split_rows(rows,split)
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]
            with Image.open(Path(root)/r['path']) as im:x=t(im.convert('RGB'))
            return x,LABELS.index(r['label'])
    loaders={s:DataLoader(Photos(s),batch_size=batch,shuffle=s=='train',num_workers=0) for s in ('train','val','test')}
    model=neural(kind,pretrained);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    loss_fn=torch.nn.CrossEntropyLoss()
    best=-1.;best_state=None;history=[]
    def evaluate(split):
        truth=[];pred=[];model.eval()
        with torch.inference_mode():
            for x,y in loaders[split]:
                z=model(x.to(device)).argmax(1).cpu().tolist()
                truth.extend([LABELS[k] for k in y.tolist()]);pred.extend([LABELS[k] for k in z])
        return report(truth,pred),truth,pred
    for epoch in range(epochs):
        model.train();total=0
        for x,y in loaders['train']:
            x=x.to(device);y=y.to(device);optimizer.zero_grad()
            loss=loss_fn(model(x),y);loss.backward();optimizer.step();total+=float(loss)*len(y)
        val,_,_=evaluate('val');history.append({'epoch':epoch+1,'loss':total/len(loaders['train'].dataset),'val_macro_f1':val['macro_f1']})
        if val['macro_f1']>best:
            best=val['macro_f1'];best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.to(device)
    metrics={}
    for split in ('val','test'):
        metrics[split],truth,pred=evaluate(split)
        with (out/f'{split}_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['truth','prediction']);w.writerows(zip(truth,pred))
    torch.save(best_state,out/'weights.pt')
    (out/'metadata.json').write_text(json.dumps({'model':kind,'labels':LABELS,'size':size,'pretrained':pretrained,'seed':seed,'n_train':len(loaders['train'].dataset),'history':history,'metrics':metrics},indent=2))
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--model',choices=['opencv','cnn','transfer'],default='opencv');ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--size',type=int,default=128);ap.add_argument('--batch',type=int,default=24);ap.add_argument('--pretrained',action='store_true')
    a=ap.parse_args();m=classical_train(a.data,a.out) if a.model=='opencv' else torch_train(a.data,a.out,a.model,a.epochs,a.batch,a.size,pretrained=a.pretrained)
    print(json.dumps(m,indent=2))
