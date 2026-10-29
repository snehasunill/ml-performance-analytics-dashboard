from pathlib import Path
import sqlite3
import pytest
from benchmark import create_demo

def test_demo_aggregates(tmp_path):
    df, csv = create_demo(tmp_path/'demo.sqlite')
    assert len(df) == 9
    assert (df.repeat_count == 5).all()
    assert csv.exists()

def test_no_duplicate_repeat(tmp_path):
    create_demo(tmp_path/'demo.sqlite')
    db = sqlite3.connect(tmp_path/'demo.sqlite')
    with pytest.raises(sqlite3.IntegrityError):
        db.execute('INSERT INTO runs(config_id,repeat,accuracy,latency_ms,throughput_images_sec) VALUES (1,1,.9,1,100)')
    db.close()
