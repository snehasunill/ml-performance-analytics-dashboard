from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from real_benchmark import init_db,export
import pytest

def test_unique_repeats_and_summary(tmp_path):
    db=tmp_path/'db.sqlite';c=init_db(db)
    c.execute('INSERT INTO configurations(model,resolution,artifact) VALUES (?,?,?)',('opencv',64,'saved'))
    cid=c.execute('SELECT id FROM configurations').fetchone()[0]
    c.execute('INSERT INTO measurements(configuration_id,repetition,accuracy,macro_f1,latency_p50_ms,latency_p95_ms,throughput_ips,failed_predictions,n_samples) VALUES (?,?,?,?,?,?,?,?,?)',(cid,1,.8,.7,12,15,30,0,25))
    with pytest.raises(Exception):
        c.execute('INSERT INTO measurements(configuration_id,repetition,accuracy,macro_f1,latency_p50_ms,latency_p95_ms,throughput_ips,failed_predictions,n_samples) VALUES (?,?,?,?,?,?,?,?,?)',(cid,1,.8,.7,12,15,30,0,25))
    c.commit();c.close();assert len(export(db,tmp_path/'result.csv'))==1
