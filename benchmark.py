"""DEMO simulated measurements. Never report them as empirical results."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCHEMA = (HERE / 'schema.sql').read_text()
QUERY = 'SELECT c.model, c.image_size, COUNT(r.run_id) AS repeat_count,\n ROUND(AVG(r.accuracy), 4) AS avg_accuracy,\n ROUND(AVG(r.latency_ms), 3) AS avg_latency_ms,\n ROUND(AVG(r.throughput_images_sec), 2) AS avg_throughput,\n ROUND(MIN(r.latency_ms), 3) AS min_latency_ms,\n ROUND(MAX(r.latency_ms), 3) AS max_latency_ms,\n SUM(r.failed_predictions) AS failed_predictions\nFROM configurations c JOIN runs r ON c.config_id=r.config_id\nGROUP BY c.model, c.image_size\nORDER BY avg_latency_ms\n'

def create_demo(path=HERE / 'demo_benchmarks.sqlite'):
    path = Path(path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    rng = np.random.default_rng(42)
    models = ['opencv_baseline','custom_cnn','transfer_model']
    sizes = [64, 128, 224]
    for mi, name in enumerate(models):
        for size in sizes:
            conn.execute('INSERT OR IGNORE INTO configurations(model,image_size) VALUES (?,?)', (name,size))
            config_id = conn.execute('SELECT config_id FROM configurations WHERE model=? AND image_size=?',(name,size)).fetchone()[0]
            for repeat in range(1,6):
                # All rows simulated to illustrate schema and downstream aggregation.
                latency = float((mi+1)*size/25+rng.uniform(.1,1.5))
                accuracy = float(np.clip(.7 + mi*.07 + (size/224)*.04 + rng.normal(0,.007),0,1))
                conn.execute('INSERT OR REPLACE INTO runs(config_id,repeat,accuracy,latency_ms,throughput_images_sec,failed_predictions) VALUES (?,?,?,?,?,?)',(config_id,repeat,accuracy,latency,1000/latency,0))
    conn.commit()
    df = pd.read_sql_query(QUERY, conn)
    assert len(df)==9 and (df.repeat_count == 5).all()
    out=path.parent/'tableau_summary.csv'
    df.to_csv(out,index=False)
    conn.close()
    return df, out

if __name__ == '__main__':
    df, out = create_demo()
    print('SIMULATED DATA — NOT REAL BENCHMARK RESULTS')
    print(df.to_string(index=False))
    print('Tableau import CSV:', out)
