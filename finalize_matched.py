import csv,json,sqlite3
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
conn=sqlite3.connect(R/'data/matched_benchmark.sqlite')
conn.row_factory=sqlite3.Row
q='''SELECT c.model,c.resolution,COUNT(*) AS repetitions,ROUND(AVG(m.accuracy),5) AS accuracy,ROUND(AVG(m.macro_f1),5) AS macro_f1,ROUND(AVG(m.latency_p50_ms),3) AS median_latency_ms,ROUND(AVG(m.latency_p95_ms),3) AS p95_latency_ms,ROUND(AVG(m.throughput_ips),3) AS throughput_ips,ROUND(MIN(m.latency_p50_ms),3) AS min_median_ms,ROUND(MAX(m.latency_p50_ms),3) AS max_median_ms,SUM(m.failed_predictions) AS failed_predictions,MIN(m.n_samples) AS min_samples,MAX(m.n_samples) AS max_samples FROM configurations c JOIN measurements m ON m.configuration_id=c.id GROUP BY c.id ORDER BY c.model,c.resolution'''
rows=[dict(r) for r in conn.execute(q)]
with (R/'reports/tableau_matched_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
rr=[dict(r) for r in conn.execute('SELECT c.model,c.resolution,m.* FROM measurements m JOIN configurations c ON c.id=m.configuration_id ORDER BY c.model,c.resolution,m.repetition')]
with (R/'reports/matched_runs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
issues=[]
if len(rows)!=9:issues.append('Expected nine configurations')
for r in rows:
 if r['repetitions']!=5:issues.append('Incomplete repeats: '+str(r))
 if r['min_samples']!=24 or r['max_samples']!=24:issues.append('Inconsistent sample count')
for r in rr:
 if any(r[k] is None for k in ['accuracy','macro_f1','latency_p50_ms','throughput_ips','failed_predictions']): issues.append('Missing measurement')
 if not (0<=r['accuracy']<=1 and 0<=r['macro_f1']<=1 and r['latency_p50_ms']>=0 and r['throughput_ips']>0):issues.append('Invalid ranges')
if len(rr)!=len({(r['configuration_id'],r['repetition']) for r in rr}):issues.append('Duplicate run keys')
audit={'configurations':len(rows),'runs':len(rr),'expected_runs':45,'sample_count_each_run':24,'failed_predictions':sum(r['failed_predictions'] for r in rows),'issues':issues,'status':'PASS' if not issues else 'FAIL'}
(R/'reports/matched_audit.json').write_text(json.dumps(audit,indent=2))
qual=[r for r in rows if r['macro_f1']>=.85 and r['repetitions']==5]
best=min(qual,key=lambda x:x['median_latency_ms'])
# One simple chart, no fabricated values
fig,ax=plt.subplots(figsize=(9,5.2))
for model in ['opencv','cnn','transfer']:
 subset=[r for r in rows if r['model']==model]
 ax.scatter([r['median_latency_ms'] for r in subset],[r['macro_f1'] for r in subset],label=model,s=80)
 for r in subset:ax.annotate(f"{r['resolution']}px",(r['median_latency_ms'],r['macro_f1']),xytext=(5,4),textcoords='offset points',fontsize=8)
ax.axhline(.85,linestyle='--',linewidth=1,label='F1 threshold = .85')
ax.set_xlabel('Average median model prediction time per image (ms)');ax.set_ylabel('Macro F1 (24 synthetic examples)');ax.set_ylim(0,1.1);ax.set_title('Matched CPU benchmark: 3 models × 3 resolutions');ax.legend();fig.tight_layout();fig.savefig(R/'reports/matched_tradeoff.png',dpi=160)
report=f'''# Matched benchmark results — experimental report

## Protocol

- Three **existing trained model artifacts**, each evaluated at 64, 128, and 224 pixels.
- Exactly **24 synthetic variants from six original COCO source photos**, with source IDs restricted to the previously held-out test split.
- All nine configurations used the identical images and labels; five repeated trials each (45 recorded runs).
- CPU-only runtime; PyTorch single inference thread and sequential per-image prediction. OpenCV Random Forest retains its artifact's thread setting.
- Per-image preprocessing/feature extraction performed **before timed inference**. Timed throughput excludes image decoding, preprocessing, API/network latency. Includes Python prediction-call overhead.
- Label classes: clean, blur, noise, jpeg. Fixed severity 0.65 and the original generator's recipes.
- Scores on 24 images are exploratory, not a substitute for the full held-out test metrics; the same 24 images are reused in all five runs. The repeats describe timing variability, not independent accuracy samples.
- Different resolutions are intentionally compared on models trained at 128 pixels; performance declines at 64/224 do not imply those architectures are inherently worse at those resolutions.

## Result

**Fastest qualifying configuration at macro F1 ≥ 0.85:** {best['model']} at {best['resolution']} px; macro F1 {best['macro_f1']:.4f}, average median prediction latency {best['median_latency_ms']:.2f} ms per image.

The full results are in `tableau_matched_summary.csv`, and every repetition is retained in `matched_runs.csv`. Audit: {audit['status']}, {audit['runs']}/{audit['expected_runs']} completed runs, {len(issues)} issues.

## Caveats

A full-size benchmark (hundreds or thousands of test images) and actual Tableau `.twb`/`.twbx` workbook have **not** yet been completed. The project now meets the 3 × 3 × 5 benchmarking structure with a small pilot sample, but its Tableau dashboard remains a separate deliverable.
'''
(R/'reports/MATCHED_BENCHMARK_REPORT.md').write_text(report)
print(audit);print('fastest',best['model'],best['resolution'])
