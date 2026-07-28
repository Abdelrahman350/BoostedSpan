#!/usr/bin/env bash
# Runs every experiment in this repo sequentially (single-GPU budget, one run at a
# time). Every config already has training.save_best_checkpoint: true -- encoder
# variants pick the best epoch via HF Trainer's native eval_strategy/
# load_best_model_at_end (train_task1.py/train_task2.py), and the QLoRA generative
# variants (qlora_allam/qlora_yehia/qlora_fanar, span_relabeler) pick it via the
# custom validation-metric callback in src/utils/best_checkpoint.py -- so every run
# below selects its checkpoint from real validation performance, never just the final
# epoch.
#
# Every training step is immediately followed by predict_eval.py against
# data/test/test_in.jsonl (the real Evaluation-phase set), reloading the checkpoint
# that step just trained -- so each experiment ends with BOTH a dev-phase submission
# (submissions/dev/) and an eval-phase submission (submissions/eval/), no separate
# manual pass needed. This includes the two k-fold configs (boosted_v2_eval.yaml,
# enhanced_track_a_v2_eval.yaml) -- predict_eval.py now knows how to reload
# k-fold-trained checkpoints and reconstruct the OOF/test-averaging path
# (run_task1_kfold_eval/run_task2_track_a_kfold_eval), which it previously couldn't.
# D9's span re-labeler (task2_span_relabeler) is the one exception: it's a val-only
# measurement script with no submission-producing code path, so it stays a plain
# training step with no eval pairing.
#
# Continues past a failed step (e.g. qlora_yehia's gated-repo block) rather than
# aborting the whole night's queue -- check logs/run_all_summary.log at the end for
# a pass/fail table, and each step's own logs/<name>.log / logs/<name>_eval.log for
# details.
#
# Usage:
#   mkdir -p logs
#   nohup bash scripts/run_all_experiments.sh > logs/run_all.log 2>&1 &
#   disown
# Launching via nohup+disown (not a bare background `&` in an interactive shell)
# matters: a backgrounded job that tries to write progress-bar output to a
# controlling terminal gets SIGTTOU-stopped by job control and silently stalls (this
# happened to a qlora_allam run earlier this session) -- nohup detaches stdio so that
# can't happen. Or just run it plain (`bash scripts/run_all_experiments.sh`) in a
# terminal you intend to keep open and watch live.

set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR="logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/run_all_summary.log"
: > "$SUMMARY"

TEAM_NAME="Nu_Analytics"
TRAINING_SETTING="both"

run_step() {
    local name="$1"
    shift
    local logfile="$LOGDIR/${name}.log"
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START $name ===" | tee -a "$SUMMARY"
    if "$@" > "$logfile" 2>&1; then
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] OK    $name (log: $logfile) ===" | tee -a "$SUMMARY"
        return 0
    else
        echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $name (log: $logfile) ===" | tee -a "$SUMMARY"
        return 1
    fi
}

# Trains via "$@", then -- only if training succeeded -- immediately reloads that
# checkpoint via predict_eval.py against test_in.jsonl to produce the eval-phase
# submission, before moving to the next experiment.
run_step_train_then_eval() {
    local name="$1" config_path="$2"
    shift 2
    if run_step "$name" "$@"; then
        run_step "${name}_eval" uv run python -m predict_eval --config "$config_path" \
            --team-name "$TEAM_NAME" --training-setting "$TRAINING_SETTING"
    else
        echo "=== SKIP  ${name}_eval (training step failed) ===" | tee -a "$SUMMARY"
    fi
}

# --- Task 1, encoder ---
run_step_train_then_eval task1_baseline   configs/task1/baseline.yaml   uv run python -m train_task1 --config configs/task1/baseline.yaml
run_step_train_then_eval task1_boosted    configs/task1/boosted.yaml    uv run python -m train_task1 --config configs/task1/boosted.yaml
run_step_train_then_eval task1_boosted_v2 configs/task1/boosted_v2.yaml uv run python -m train_task1 --config configs/task1/boosted_v2.yaml

# --- Task 1, QLoRA generative ---
run_step_train_then_eval task1_qlora_allam configs/task1/qlora_allam.yaml uv run python -m train_task1_generative --config configs/task1/qlora_allam.yaml
run_step_train_then_eval task1_qlora_fanar configs/task1/qlora_fanar.yaml uv run python -m train_task1_generative --config configs/task1/qlora_fanar.yaml
# qlora_yehia requires a granted HF access request (Navid-AI/Yehia-7B-preview is
# gated) + `huggingface-cli login`/HF_TOKEN. Commented out -- uncomment once access is
# sorted; it'll then train and produce both submissions like everything else here.
# run_step_train_then_eval task1_qlora_yehia configs/task1/qlora_yehia.yaml uv run python -m train_task1_generative --config configs/task1/qlora_yehia.yaml

# --- Task 2, encoder ---
run_step_train_then_eval task2_baseline            configs/task2/baseline.yaml            uv run python -m train_task2 --config configs/task2/baseline.yaml
run_step_train_then_eval task2_boosted_crf         configs/task2/boosted_crf.yaml         uv run python -m train_task2 --config configs/task2/boosted_crf.yaml
run_step_train_then_eval task2_enhanced_track_a    configs/task2/enhanced_track_a.yaml    uv run python -m train_task2 --config configs/task2/enhanced_track_a.yaml
run_step_train_then_eval task2_enhanced_track_a_v2 configs/task2/enhanced_track_a_v2.yaml uv run python -m train_task2 --config configs/task2/enhanced_track_a_v2.yaml
run_step_train_then_eval task2_enhanced_track_b    configs/task2/enhanced_track_b.yaml    uv run python -m train_task2 --config configs/task2/enhanced_track_b.yaml
run_step_train_then_eval task2_span_scorer         configs/task2/span_scorer.yaml         uv run python -m train_task2 --config configs/task2/span_scorer.yaml

# --- Task 2, D9 span re-labeler hybrid (separate script, not train_task2.py) ---
# No eval pairing: this is a val-only measurement script (trains an adapter, sweeps
# confidence thresholds against enhanced_track_a's val predictions) with no
# submission-producing code path today.
run_step task2_span_relabeler uv run python scripts/d9_train_and_eval_relabeler.py

# --- Eval-phase k-fold configs (train_on_dev_refs + num_folds) ---
# No longer gated behind RUN_EVAL_PHASE: predict_eval.py now supports reloading
# k-fold-trained checkpoints, so these get the same train-then-eval pairing as
# everything else. Their own dev-phase prediction is always suppressed by design
# (train_task1.py/train_task2.py's skip_submission guard -- a model trained on dev
# labels can't legally produce a dev-phase submission), so the _eval half below is
# the only real submission these two ever produce.
run_step_train_then_eval task1_boosted_v2_eval    configs/task1/boosted_v2_eval.yaml    uv run python -m train_task1 --config configs/task1/boosted_v2_eval.yaml
run_step_train_then_eval task2_enhanced_track_a_v2_eval configs/task2/enhanced_track_a_v2_eval.yaml uv run python -m train_task2 --config configs/task2/enhanced_track_a_v2_eval.yaml

echo
echo "All steps finished. Summary:"
cat "$SUMMARY"
