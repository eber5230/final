# Quickstart 256

Submission-like baseline check on a cheap horizon:

```bash
cd /home/richard/gb10-llm-stack/parameter-golf/records/track_non_record_16mb/2026-04-05_submission_like_baseline_v1

RUN_ID=baseline_256_submissionlike_v1 \
DATA_PATH=/home/richard/gb10-llm-stack/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/home/richard/gb10-llm-stack/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
ITERATIONS=256 \
VAL_LOSS_EVERY=64 \
SKIP_INITIAL_VAL=1 \
MAX_WALLCLOCK_SECONDS=0 \
WARMDOWN_ITERS=0 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Main values to inspect at the end:

- `final_int6_roundtrip`
- `final_int6_sliding_window`
- `final_int6_sliding_window_s64`
- `Total submission size int6+zstd`
