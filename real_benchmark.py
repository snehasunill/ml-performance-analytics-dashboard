"""CPU inference timing against real saved artifacts and real labeled images.
Requires 02_image_quality dependencies. No simulated results.
"""
import argparse,csv,json,sqlite3,sys,time,statistics
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
IMAGE_PROJECT=HERE/'02_image_quality'
sys.path.insert(0,str(IMAGE_PROJECT))
from quality.data import LABELS
from quality.models import features,neural,transform
import joblib

def init_db(db):
    conn=sqlite3.connect(db)
    conn.execute('PRAGMA foreign_keys=ON')
    conn.executescript("""CREATE TABLE IF NOT EXISTS configurations (
      id INTEGER PRIMARY KEY, model TEXT NOT NULL, resolution INTEGER NOT NULL,
      artifact TEXT NOT NULL, UNIQUE(model,resolution,artifact));
      CREATE TABLE IF NOT EXISTS measurements (
      configuration_id INTEGER NOT NULL REFERENCES configurations(id),
      repetition INTEGER NOT NULL CHECK(repetition>0),
      accuracy REAL NOT NULL CHECK(accuracy BETWEEN 0 AND 1),
      macro_f1 REAL NOT NULL CHECK(macro_f1 BETWEEN 0 AND 1),
      latency_p50_ms REAL NOT NULL CHECK(latency_p50_ms>=0),
      latency_p95_ms REAL NOT NULL CHECK(latency_p95_ms>=0),
      throughput_ips REAL NOT NULL CHECK(throughput_ips>=0),
      failed_predictions INTEGER NOT NULL CHECK(failed_predictions>=0),
      n_samples INTEGER NOT NULL CHECK(n_samples>0),
      measured_at TEXT DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(configuration_id,repetition));""")
    return conn

def load_predictor(artifact,size):
    meta=json.loads((Path(artifact)/'metadata.json').read_text())
    kind=meta['model']
    if kind=='opencv':
        model=joblib.load(Path(artifact)/'model.joblib')
        def predict(im):return str(model.predict([features(im.resize((size,size)))])[0])
    else:
        import torch
        torch.set_num_threads(1)
        model=neural(kind,False)
        model.load_state_dict(torch.load(Path(artifact)/'weights.pt',map_location='cpu',weights_only=True))
        model.eval();tx=transform(size,kind)
        def predict(im):
            with torch.inference_mode():return LABELS[int(model(tx(im).unsqueeze(0)).argmax(1).item())]
    return kind,predict

def measure(artifact,size,samples,warmup=4):
    from sklearn.metrics import accuracy_score,f1_score
    kind,predict=load_predictor(artifact,size)
    if not samples:raise ValueError('no samples')
    for i in range(warmup):predict(samples[i%len(samples)][0])
    lat=[];labels=[];pred=[];failed=0
    start=time.perf_counter()
    for im,label in samples:
        t=time.perf_counter()
        try: p=predict(im)
        except Exception:p='__ERROR__';failed+=1
        lat.append((time.perf_counter()-t)*1000)
        labels.append(label);pred.append(p)
    duration=time.perf_counter()-start
    return {'accuracy':accuracy_score(labels,pred),
      'macro_f1':f1_score(labels,pred,labels=list(LABELS),average='macro',zero_division=0),
      'latency_p50_ms':float(np.percentile(lat,50)),'latency_p95_ms':float(np.percentile(lat,95)),
      'throughput_ips':len(samples)/duration,'failed_predictions':failed,'n_samples':len(samples)}

def read_test_samples(dataset,max_samples=400):
    rows=list(csv.DictReader((Path(dataset)/'manifest.csv').open()))
    samples=[]
    for r in rows:
        if r['split']!='test':continue
        with Image.open(Path(dataset)/r['path']) as im:samples.append((im.convert('RGB').copy(),r['label']))
        if len(samples)>=max_samples:break
    if not samples:raise ValueError('no held-out samples')
    return samples

def run(db,dataset,artifacts,sizes=(64,128,224),repetitions=5,max_samples=400):
    if repetitions<1:raise ValueError('repetitions must be positive')
    samples=read_test_samples(dataset,max_samples)
    conn=init_db(db)
    for artifact in artifacts:
        for size in sizes:
            kind,_=load_predictor(artifact,size)
            conn.execute('INSERT OR IGNORE INTO configurations(model,resolution,artifact) VALUES (?,?,?)',(kind,size,str(artifact)))
            config_id=conn.execute('SELECT id FROM configurations WHERE model=? AND resolution=? AND artifact=?',(kind,size,str(artifact))).fetchone()[0]
            for rep in range(1,repetitions+1):
                result=measure(artifact,size,samples)
                conn.execute("""INSERT OR REPLACE INTO measurements(configuration_id,repetition,accuracy,macro_f1,latency_p50_ms,latency_p95_ms,throughput_ips,failed_predictions,n_samples)
                VALUES (?,?,?,?,?,?,?,?,?)""",(config_id,rep,*result.values()))
                conn.commit();print(kind,size,rep,result)
    conn.close()

def export(db,out,accuracy_threshold=.85):
    conn=sqlite3.connect(db)
    query="""SELECT c.model,c.resolution,c.artifact,COUNT(*) repeats,
        AVG(m.accuracy) avg_accuracy,AVG(m.macro_f1) avg_macro_f1,
        AVG(m.latency_p50_ms) avg_p50_ms,AVG(m.latency_p95_ms) avg_p95_ms,
        AVG(m.throughput_ips) avg_throughput_ips,SUM(m.failed_predictions) failed_predictions,
        MIN(m.latency_p50_ms) min_p50_ms,MAX(m.latency_p50_ms) max_p50_ms
        FROM configurations c JOIN measurements m ON m.configuration_id=c.id
        GROUP BY c.id ORDER BY avg_p50_ms ASC"""
    rows=conn.execute(query).fetchall();names=[x[0] for x in conn.execute(query).description]
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.writer(f);w.writerow(names);w.writerows(rows)
    eligible=[dict(zip(names,row)) for row in rows if row[names.index('avg_macro_f1')]>=accuracy_threshold]
    print('Fastest meeting F1 threshold:',eligible[0] if eligible else 'NONE')
    conn.close();return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--db',default=str(HERE/'results.sqlite'))
    ap.add_argument('--opencv',required=True);ap.add_argument('--cnn',required=True);ap.add_argument('--transfer',required=True)
    ap.add_argument('--runs',type=int,default=5);ap.add_argument('--threshold',type=float,default=.85)
    a=ap.parse_args();run(a.db,a.data,[a.opencv,a.cnn,a.transfer],repetitions=a.runs)
    export(a.db,HERE/'tableau_real_results.csv',a.threshold)
