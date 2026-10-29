# Benchmark methodology and limitations

- 3 saved models × 3 resolutions (64, 128, 224) × 5 repeated runs = **45 recorded runs**.
- All configurations used the **same 24 synthetic images from six held-out COCO source photographs**.
- The five repeats measure runtime variability; they are **not independent accuracy samples**.
- Model inference time excludes preprocessing, decoding and HTTP/network overhead. Results are specific to the CPU environment.
- Original neural models were trained at 128 px; changing input size at evaluation time is not retraining.
- A small fixed synthetic sample does not support general claims about production accuracy or performance.
- At macro F1 ≥ 0.85, **CNN, 128 px** had the lowest measured median latency (approximately 7.57 ms per image).

See `reports/matched_runs.csv` for all raw measurements and `reports/matched_audit.json` for completeness checks.
