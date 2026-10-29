"""Audit measured benchmark runs and export only verified, complete configurations."""
import argparse
import csv
import json
import sqlite3
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS configurations (
  id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  resolution INTEGER NOT NULL,
  artifact TEXT NOT NULL,
  UNIQUE(model,resolution,artifact)
);
CREATE TABLE IF NOT EXISTS measurements (
  configuration_id INTEGER NOT NULL REFERENCES configurations(id),
  repetition INTEGER NOT NULL CHECK(repetition BETWEEN 1 AND 5),
  accuracy REAL NOT NULL CHECK(accuracy BETWEEN 0 AND 1),
  macro_f1 REAL NOT NULL CHECK(macro_f1 BETWEEN 0 AND 1),
  latency_p50_ms REAL NOT NULL CHECK(latency_p50_ms >= 0),
  latency_p95_ms REAL NOT NULL CHECK(latency_p95_ms >= 0),
  throughput_ips REAL NOT NULL CHECK(throughput_ips > 0),
  failed_predictions INTEGER NOT NULL CHECK(failed_predictions >= 0),
  n_samples INTEGER NOT NULL CHECK(n_samples > 0),
  measured_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(configuration_id,repetition)
);
'''

def audit(db_path, output_dir, threshold=0.85):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute('PRAGMA foreign_keys=ON')
    config_rows = con.execute('SELECT id,model,resolution,artifact FROM configurations ORDER BY model,resolution').fetchall()
    measurement_cols = [d[1] for d in con.execute('PRAGMA table_info(measurements)')]
    details = []
    summary = []
    issues = []
    for cid, model, resolution, artifact in config_rows:
        rows = [dict(zip(measurement_cols, r)) for r in con.execute('SELECT * FROM measurements WHERE configuration_id=? ORDER BY repetition',(cid,)).fetchall()]
        repetitions = [r['repetition'] for r in rows]
        complete = repetitions == [1,2,3,4,5]
        if not complete:
            issues.append(f'{model}/{resolution}: missing or unexpected repetitions {repetitions}')
        sample_sizes = {r['n_samples'] for r in rows}
        if len(sample_sizes) > 1:
            issues.append(f'{model}/{resolution}: inconsistent sample sizes')
        for r in rows:
            details.append({'model':model, 'resolution':resolution, 'artifact':artifact, **r})
        if rows:
            mean = lambda key: sum(r[key] for r in rows)/len(rows)
            summary.append({'model':model,'resolution':resolution,'run_count':len(rows), 'complete':complete,
              'n_samples': next(iter(sample_sizes)), 'accuracy':mean('accuracy'),'macro_f1':mean('macro_f1'),
              'p50_ms':mean('latency_p50_ms'), 'p95_ms':mean('latency_p95_ms'),
              'throughput_ips':mean('throughput_ips'), 'failed_predictions':sum(r['failed_predictions'] for r in rows),
              'p50_range_ms':max(r['latency_p50_ms'] for r in rows)-min(r['latency_p50_ms'] for r in rows)})
    for filename, values in [('benchmark_runs.csv',details),('tableau_summary.csv',summary)]:
        if values:
            with open(output_dir/filename,'w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    # Complete and meeting accuracy threshold, with no failed predictions.
    eligible=[r for r in summary if r['complete'] and r['macro_f1']>=threshold and r['failed_predictions']==0]
    best=min(eligible,key=lambda r:r['p50_ms']) if eligible else None
    report={'configurations_present':len(summary),'runs_present':len(details),'target_configurations':9,
        'target_runs':45,'issues':issues,'fastest_eligible':best,'threshold_metric':'macro_f1',
        'threshold':threshold, 'scope':'partial measured OpenCV CPU benchmark; not three-model comparison'}
    (output_dir/'audit.json').write_text(json.dumps(report,indent=2))
    con.close()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--database',default='data/verified_opencv.sqlite')
    p.add_argument('--out',default='reports')
    p.add_argument('--threshold',type=float,default=0.85)
    args=p.parse_args()
    print(json.dumps(audit(args.database,args.out,args.threshold),indent=2))
