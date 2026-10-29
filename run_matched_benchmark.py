"""Run matched 3-model x 3-resolution x 5-repeat CPU experiment on held-out source photos."""
import csv,io,json,random,sqlite3,time,zipfile,sys
from pathlib import Path
from statistics import median
import numpy as np
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,f1_score
import torch, joblib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'02_image_quality'))
from quality.data import LABELS,corrupt
from quality.models import neural,transform,features
DB=ROOT/'data'/'matched_benchmark.sqlite'
SOURCES=Path('/mnt/data/selected_2000.zip')
MANIFEST=Path(ROOT/'data'/'full_manifest.csv')
ART=ROOT/'artifacts'

def samples_from_zip(count=6):
    rows=list(csv.DictReader(MANIFEST.open()))
    ids=sorted({r['source_id'] for r in rows if r['split']=='test'})[:count]
    z=zipfile.ZipFile(SOURCES)
    out=[]
    for i,sid in enumerate(ids):
        with z.open('selected_2000/'+Path(sid).name) as f:
            base=ImageOps.fit(Image.open(io.BytesIO(f.read())).convert('RGB'),(128,128))
        for j,label in enumerate(LABELS):
            out.append((corrupt(base,label,42+i*17+j,.65),label,sid))
    assert len(set(s for _,_,s in out))==count
    return out

def load_models():
    rf=joblib.load(ART/'opencv'/'model.joblib')
    models={'opencv':rf}
    for kind in ('cnn','transfer'):
        model=neural(kind,False)
        model.load_state_dict(torch.load(ART/kind/'weights.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval()
        models[kind]=model
    return models

def main():
    torch.set_num_threads(1)
    samples=samples_from_zip()
    models=load_models()
    conn=sqlite3.connect(DB)
    conn.executescript('''PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS configurations (id INTEGER PRIMARY KEY,model TEXT NOT NULL,resolution INTEGER NOT NULL, UNIQUE(model,resolution));
    CREATE TABLE IF NOT EXISTS measurements (configuration_id INTEGER NOT NULL REFERENCES configurations(id),repetition INTEGER NOT NULL,accuracy REAL NOT NULL,macro_f1 REAL NOT NULL,latency_p50_ms REAL NOT NULL,latency_p95_ms REAL NOT NULL,throughput_ips REAL NOT NULL,failed_predictions INTEGER NOT NULL,n_samples INTEGER NOT NULL,PRIMARY KEY(configuration_id,repetition));''')
    for kind, model in models.items():
        for size in (64,128,224):
            conn.execute('INSERT OR IGNORE INTO configurations(model,resolution) VALUES (?,?)',(kind,size))
            cid=conn.execute('SELECT id FROM configurations WHERE model=? AND resolution=?',(kind,size)).fetchone()[0]
            tx=transform(size,kind) if kind!='opencv' else None
            inputs=[features(im.resize((size,size))) if kind=='opencv' else tx(im) for im,_,_ in samples]
            truth=[label for _,label,_ in samples]
            def pred(x):
                if kind=='opencv': return str(model.predict([x])[0])
                with torch.inference_mode():return LABELS[int(model(x.unsqueeze(0)).argmax(1))]
            for x in inputs[:2]:pred(x)
            for rep in range(1,6):
                if conn.execute('SELECT 1 FROM measurements WHERE configuration_id=? AND repetition=?',(cid,rep)).fetchone():continue
                y=[];times=[];failed=0;start=time.perf_counter()
                for x in inputs:
                    t=time.perf_counter()
                    try:y.append(pred(x))
                    except Exception:y.append('__ERROR__');failed+=1
                    times.append((time.perf_counter()-t)*1000)
                duration=time.perf_counter()-start
                vals=(accuracy_score(truth,y),f1_score(truth,y,labels=list(LABELS),average='macro',zero_division=0),float(np.percentile(times,50)),float(np.percentile(times,95)),len(inputs)/duration,failed,len(inputs))
                conn.execute('INSERT INTO measurements VALUES (?,?,?,?,?,?,?,?,?)',(cid,rep,*vals));conn.commit()
                print(kind,size,rep,'macroF1',round(vals[1],4),'p50',round(vals[2],2),flush=True)
    conn.close()
    print('FINISHED',DB)
if __name__=='__main__':main()
