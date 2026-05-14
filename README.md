# MRVP: Motion-Reset Viability Programming

This repository implements **MRVP: Motion-Reset Viability Programming**. Given a root driving scene and a library of candidate emergency prefixes, **CMRT** predicts a distribution over action-induced motion-reset recovery problems. **RPFN** synthesizes closed-loop recovery programs with funnel certificates for each reset sample. The final action is selected by **lower-tail CVaR** of the best program certificates inside the harm-comparable candidate set.

```text
candidate prefix a
  -> CMRT: action-conditioned motion-reset problem distribution
       R_a = (reset_state, reset_slots, recovery_world, degradation, reset_uncertainty)
  -> RPFN: closed-loop recovery programs with funnel certificates
       P_a^ell = (policy, target, funnel)
  -> degraded dynamics rollout
       c_a^{m,ell} = min certificate margin
  -> S(a) = LCVaR_beta({ max_ell c_a^{m,ell} }_m)
  -> select argmax S(a) among harm-comparable candidates
```

## Repository layout

```text
mrvp/
  data/
    schema.py              # reset-problem schema + legacy aliases
    dataset.py             # root-level dataset and collate utilities
    reset_targets.py       # lightweight reset-boundary target construction
    synthetic.py           # simulator-free smoke-test data
    metadrive_rollout.py   # MetaDrive/light2d counterfactual rollouts
  dynamics/
    degraded_bicycle.py    # differentiable degraded rollout f_d
  safety/
    margins.py             # road/sec/stab/ctrl/goal margins
    certificates.py        # funnel + margin certificate aggregation
  teachers/
    degraded_mpc.py        # lightweight degraded shooting-MPC teacher
    cem_recovery.py        # CEM-style recovery teacher interface
  models/
    cmrt.py                # Counterfactual Motion Reset Tokenizer
    rpfn.py                # Recovery Program-Funnel Network
    reset_memory.py        # root/action memory and reset slot layers
    recovery_programs.py   # program queries, policy and funnel heads
    msrt.py                # deprecated compatibility shim
    rpn.py                 # deprecated compatibility shim
  action_selection.py      # tail-consistent lower-CVaR selector
  selection.py             # deprecated compatibility shim
scripts/
  generate_synthetic.py
  generate_metadrive.py
  train.py
  evaluate_predicted_mrvp.py
  run_select.py
  validate_schema_no_leakage.py
  validate_counterfactual_roots.py
  diagnose_metadrive_dataset_quality.py
  validate_method_alignment.py
```

## Install

```bash
pip install -e .
pip install -r requirements.txt
```

Optional simulator dependencies such as `metadrive-simulator` are only needed for simulator rollouts. The synthetic/light2d path runs with PyTorch and NumPy. A dependency-free degraded shooting-MPC/CEM teacher interface is provided under `mrvp/teachers/` for stronger recovery-label generation than the fastest heuristic smoke path.

## One-command smoke workflow

```bash
python scripts/generate_synthetic.py \
  --output data/synthetic_mrvp.jsonl \
  --n-roots 240 \
  --seed 7

python scripts/train.py \
  --data data/synthetic_mrvp.jsonl \
  --out-dir runs/synthetic \
  --stage all \
  --epochs 3 \
  --batch-size 128 \
  --device auto

python scripts/evaluate_predicted_mrvp.py \
  --data data/synthetic_mrvp.jsonl \
  --cmrt runs/synthetic/cmrt.pt \
  --rpfn runs/synthetic/rpfn_finetuned.pt \
  --split test \
  --num-reset-samples 8 \
  --beta 0.2
```

Training writes:

```text
runs/.../cmrt.pt
runs/.../rpfn.pt
runs/.../rpfn_finetuned.pt
runs/.../config_resolved.yaml
```

## Reset-problem JSONL schema

Each row is one candidate action under one root scene. All actions with the same `root_id` must share the same root observation and split.

| Field | Meaning |
|---|---|
| `root_id`, `split`, `family` | Root scenario identity and split. |
| `action_id`, `action_name`, `action_vec` | Candidate emergency prefix. Control convention is `[delta, throttle, brake, duration]` for the prefix. |
| `o_hist`, `h_ctx`, `x_t` | Root-scene observation, context and ego state before the candidate prefix. |
| `rho_imp`, `harm_bin` | First-impact harm gate. Selection only compares candidates in the minimum harm bin for the root. |
| `reset_time`, `reset_state` | Recovery-reasoning boundary target. Synthetic/light2d uses a lightweight difficulty target instead of always taking prefix end. |
| `degradation` | Degraded control/dynamics vector. |
| `reset_uncertainty_target` | Optional target for reset uncertainty diagnostics. |
| `recovery_world` | Structured `A/O/G/Y` recovery world: `affordance`, `occupancy`, `goal`, `actor_response`. |
| `teacher_u`, `teacher_traj` | Teacher recovery controls and trajectory for RPFN execution loss. Control convention is `[delta, F_b, F_x]`. |
| `m_star`, `b_star`, `s_star` | Teacher certificate profile, active bottleneck and scalar best certificate proxy. |
| `audit` | Optional diagnostics such as event type/side. Not used by RPFN or selection. |

The loader returns these main tensor keys:

```python
batch["reset_state"]
batch["reset_time"]
batch["degradation"]
batch["recovery_world_vec"]
batch["reset_slots"]
batch["reset_slots_target"]      # optional distillation/ablation only
batch["audit_event_type_id"]     # optional diagnostic only
```

### Backward compatibility

Legacy rows are normalized as aliases:

```python
reset_state = row.get("reset_state", row.get("r_reset", row.get("x_plus")))
reset_time = row.get("reset_time", row.get("event_time", row.get("tau")))
degradation = row.get("degradation", row.get("deg", row.get("d_deg")))
recovery_world = row.get("recovery_world", row.get("world_plus", row.get("w_plus")))
```

The batch still includes `x_plus`, `event_time`, `deg`, `world_plus`, and `event_tokens` for compatibility. New training losses do not supervise latent reset slots from legacy `event_tokens` unless `--lambda-slot-distill` is explicitly set.

## Train CMRT and RPFN

```bash
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage all \
  --epochs 30 \
  --batch-size 256 \
  --device auto \
  --reset-slot-count 16 \
  --reset-slot-dim 64 \
  --program-count 6
```

Stages:

```text
--stage cmrt      # train CounterfactualMotionResetTokenizer
--stage rpfn      # train RecoveryProgramFunnelNetwork
--stage finetune  # same-root order fine-tuning for RPFN
--stage all       # run all stages
```

Important loss defaults:

```text
CMRT: reset=1.0, world=0.5, degradation=1.0, counterfactual=0.2, uncertainty=1.0, audit_event=0.0, slot_distill=0.0
RPFN: execution=0.5, certificate=1.0, funnel=0.2, order=1.0
```

`audit_event` is off by default because event type is diagnostic. Calibration is not required by the default selection path; optional certificate correction can be added as an experiment.

## Evaluate predicted MRVP

```bash
python scripts/evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
  --cmrt runs/metadrive_mrvp/cmrt.pt \
  --rpfn runs/metadrive_mrvp/rpfn_finetuned.pt \
  --split test \
  --num-reset-samples 16 \
  --beta 0.2 \
  --output runs/metadrive_mrvp/eval_predicted_test.json
```

Primary reported methods include:

```text
Predicted_MRVP_CMRT_RPFN_LCVaR
Predicted_MRVP_CMRT_RPFN_mean_certificate
Severity_only
Weighted_post_impact_cost
Teacher_oracle
```

## Run one-root action selection

```bash
python scripts/run_select.py \
  --data data/metadrive_mrvp.jsonl \
  --root-id <root_id> \
  --split test \
  --cmrt runs/metadrive_mrvp/cmrt.pt \
  --rpfn runs/metadrive_mrvp/rpfn_finetuned.pt \
  --num-reset-samples 16 \
  --beta 0.2
```

The output contains:

```json
{
  "selected_action_id": 0,
  "selected_local_index": 0,
  "score": 0.12,
  "admissible_indices": [0, 1, 2],
  "tail_sample_index": 3,
  "program_index": 2,
  "candidate_summaries": []
}
```

## Dataset validation

```bash
python scripts/validate_counterfactual_roots.py \
  --data data/metadrive_mrvp.jsonl \
  --expected-actions 8

python scripts/validate_schema_no_leakage.py \
  --data data/metadrive_mrvp.jsonl \
  --fail-on-leakage

python scripts/diagnose_metadrive_dataset_quality.py \
  --data data/metadrive_mrvp.jsonl \
  --out runs/metadrive_mrvp/dataset_quality.json

python scripts/validate_method_alignment.py --fail-on-error
```

Validation checks include root-level split consistency, action count per root, `x_t/h_ctx/o_hist` counterfactual consistency, reset-time distribution, finite reset states, `m_star` distribution, negative-certificate rate, harm-bin distribution, and recovery-world leakage warnings.

## Full reproduction-style flow

```bash
# 1. Generate counterfactual root-level dataset
python scripts/generate_metadrive.py \
  --output data/metadrive_mrvp.jsonl \
  --n-roots 1000 \
  --backend metadrive \
  --seed 13

# 2. Validate dataset construction
python scripts/validate_counterfactual_roots.py \
  --data data/metadrive_mrvp.jsonl

python scripts/validate_schema_no_leakage.py \
  --data data/metadrive_mrvp.jsonl

# 3. Train CMRT + RPFN
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage all \
  --epochs 30 \
  --batch-size 256 \
  --device auto

# 4. Evaluate full predicted MRVP chain
python scripts/evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
  --cmrt runs/metadrive_mrvp/cmrt.pt \
  --rpfn runs/metadrive_mrvp/rpfn_finetuned.pt \
  --split test \
  --num-reset-samples 16 \
  --beta 0.2 \
  --output runs/metadrive_mrvp/eval_predicted_test.json

# 5. Run one root action selection
python scripts/run_select.py \
  --data data/metadrive_mrvp.jsonl \
  --root-id <root_id> \
  --split test \
  --cmrt runs/metadrive_mrvp/cmrt.pt \
  --rpfn runs/metadrive_mrvp/rpfn_finetuned.pt \
  --num-reset-samples 16 \
  --beta 0.2
```

## Compatibility notes

`mrvp/models/msrt.py`, `mrvp/models/rpn.py`, and `mrvp/selection.py` remain as deprecated shims. They exist to keep old imports and checkpoints loadable, but the documented path uses CMRT, RPFN and tail-consistent lower-CVaR selection.
