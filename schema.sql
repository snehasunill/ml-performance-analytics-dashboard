PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS configurations (
  config_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  image_size INTEGER NOT NULL,
  UNIQUE(model, image_size)
);
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY,
  config_id INTEGER NOT NULL REFERENCES configurations(config_id),
  repeat INTEGER NOT NULL CHECK(repeat BETWEEN 1 AND 5),
  accuracy REAL NOT NULL CHECK(accuracy BETWEEN 0 AND 1),
  latency_ms REAL NOT NULL CHECK(latency_ms > 0),
  throughput_images_sec REAL NOT NULL CHECK(throughput_images_sec > 0),
  failed_predictions INTEGER NOT NULL DEFAULT 0 CHECK(failed_predictions >= 0),
  UNIQUE(config_id, repeat)
);
