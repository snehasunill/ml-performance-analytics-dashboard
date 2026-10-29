import sqlite3
from analyze_results import audit
from real_benchmark import init_db

def test_audit_detects_incomplete_and_no_fabricated_summary(tmp_path):
    db = tmp_path / 'bench.sqlite'
    con = init_db(db)
    con.execute('INSERT INTO configurations(model,resolution,artifact) VALUES (?,?,?)',('opencv',128,'test'))
    cid = con.execute('SELECT id FROM configurations').fetchone()[0]
    con.execute('INSERT INTO measurements(configuration_id,repetition,accuracy,macro_f1,latency_p50_ms,latency_p95_ms,throughput_ips,failed_predictions,n_samples) VALUES (?,?,?,?,?,?,?,?,?)', (cid,1,.9,.9,10,20,100,0,32))
    con.commit();con.close()
    result=audit(db,tmp_path/'reports')
    assert result['runs_present']==1
    assert result['fastest_eligible'] is None
    assert result['issues']
