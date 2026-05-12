# MRVP PyTorch 实现（clean event-token / strategy-verifier 版本）

本版本把代码路径收紧为：

```text
pre-impact scene + candidate action
  -> MSRT: Action-conditioned Recovery Event Tokenizer
     输出 event_type/event_time, x_plus distribution, learned event_tokens, clean world_plus, deg
  -> RPN: Policy-conditioned Recovery Viability Network
     输入 MSRT event_tokens + clean world_plus + x_plus + deg
     解码 recovery controls，经 degraded bicycle dynamics rollout 得到 strategy_traj
     基于 strategy_traj + world_plus + deg 验证五个 strategy-conditioned signed margins
  -> calibration + CVaR action selection under min-harm-bin gate
```

`z_mech` / `audit_mech` 仅用于审计、probe、兼容旧数据，**不会再 fallback 成 `event_tokens`，也不会作为新版 RPN 主输入**。

---

## 1. 安装

```bash
cd MRVP-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## 2. 数据 schema 的关键约束

推荐状态 layout：

```text
[p_x, p_y, psi, v_x, v_y, yaw_rate, a_x, a_y, beta, delta, F_b, F_x]
```

推荐控制 layout：

```text
[delta, F_b, F_x]
```

重要字段：

| 字段 | 说明 |
|---|---|
| `event_tokens` | learned token；缺失时 loader 输出全零，并设置 `has_event_tokens=0` |
| `audit_mech` / `z_mech` | 审计/probe；不进入 RPN 主路径 |
| `world_plus` | 只能包含 post-event observable/predictable world，不得来自 recovery teacher trajectory 或 `m_star/s_star` |
| `m_star`, `b_star`, `s_star` | teacher label source，只能作为训练/评估 target |
| `is_harm_admissible` | root 内最小 harm bin actions，用于 pair mining 和 selection |

旧字段仍兼容：`d_deg -> deg`，`r_star -> m_star`，`z_mech -> audit_mech`。但旧 `z_mech` 不再作为 event token fallback。

---

## 3. 生成 clean light2d/MetaDrive schema 数据

无 simulator smoke 数据：

```bash
python scripts/generate_synthetic.py \
  --output data/synthetic_mrvp.jsonl \
  --n-roots 240 \
  --seed 7
```

MetaDrive/light2d 统一入口：

```bash
python scripts/generate_metadrive.py \
  --output data/metadrive_mrvp_clean.jsonl \
  --n-roots 1000 \
  --backend auto \
  --seed 13
```

当前 zip 内置 clean light2d fallback；`world_plus` 不再由 `recovery_traj`、`teacher_u`、`m_star/r_star/s_star` 构造。

---

## 4. 数据质量与 no-leakage 检查

Schema leakage 检查：

```bash
python scripts/validate_schema_no_leakage.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --output runs/metadrive_mrvp/schema_no_leakage.json
```

数据质量报告：

```bash
python scripts/diagnose_metadrive_dataset_quality.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out runs/metadrive_mrvp/dataset_quality_report.json \
  --eps-s 0.25 \
  --expected-actions 8
```

关键通过标准建议：

```text
root_split_leak_count = 0
audit_token_fallback_rows = 0
selectable_root_rate_adm_ge_2 >= 0.50
informative_root_rate_adm_pair_eps >= 0.20
recoverable_row_rate in [0.05, 0.95]
```

---

## 5. 训练

### 5.1 一条命令跑 clean pipeline

```bash
python scripts/train.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage all \
  --epochs 30 \
  --batch-size 256 \
  --device auto \
  --torch-threads 1
```

`--stage all` 依次运行：

```text
msrt -> rpn_oracle_clean -> pair
```

保存：

```text
runs/metadrive_mrvp/msrt.pt
runs/metadrive_mrvp/rpn.pt
runs/metadrive_mrvp/rpn_finetuned.pt
```

### 5.2 分阶段训练

MSRT：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage msrt \
  --epochs 30 \
  --batch-size 256 \
  --lambda-token 0.0 \
  --device auto \
  --torch-threads 1
```

RPN oracle-clean：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage rpn_oracle_clean \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --epochs 30 \
  --batch-size 256 \
  --strategy-count 6 \
  --recovery-horizon 30 \
  --lambda-optimism 0.2 \
  --lambda-diversity 0.05 \
  --device auto \
  --torch-threads 1
```

RPN 训练时不会直接读取 dataset `event_tokens`；它使用：

```text
true x_plus + clean world_plus + true deg + frozen MSRT-generated event_tokens
```

Pair fine-tuning：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage pair \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn.pt \
  --epochs 10 \
  --batch-size 128 \
  --eps-s 0.25 \
  --pair-margin 0.05 \
  --lambda-ord 1.0 \
  --device auto \
  --torch-threads 1
```

---

## 6. Calibration

```bash
python scripts/calibrate.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration-mode oracle_clean \
  --output runs/metadrive_mrvp/calibration.json \
  --split cal \
  --delta-b 0.02 \
  --n-min 20 \
  --device auto \
  --torch-threads 1
```

Calibration 默认使用 clean oracle 输入：true transition/world/deg + MSRT-generated tokens。

---

## 7. 评估

### 7.1 Clean oracle true-transition

```bash
python scripts/evaluate.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --split test \
  --output runs/metadrive_mrvp/eval_oracle_clean_test.json \
  --device auto \
  --torch-threads 1
```

输出方法名包括：

```text
Oracle_clean_true_transition
Oracle_clean_true_transition_uncalibrated
Severity_only
Weighted_post_impact_audit_debug
Teacher_oracle
```

如需检查旧 leaky 路径，显式添加：

```bash
--include-leaky-oracle-debug
```

并只报告为 `Oracle_leaky_do_not_report`。

### 7.2 Predicted MRVP 真推理链路

```bash
python scripts/evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --split test \
  --num-samples 16 \
  --beta 0.2 \
  --output runs/metadrive_mrvp/eval_predicted_test.json \
  --device auto \
  --torch-threads 1
```

关键输出：

```text
Teacher_oracle
Oracle_clean_true_transition
Predicted_MRVP_MSRT_CVaR
Predicted_MRVP_MSRT_mean_risk
Severity_only
Weighted_post_impact_audit_debug
```

核心指标包括：

```text
selected_s_star_mean
selected_recoverable_rate
same_harm_pair_accuracy
shift_regret
mean_violation_depth
first_impact_harm_violation_rate
envelope_violation_rate
coverage
frr
```

`envelope_violation_rate` 应为 0，否则 harm gate 被破坏。

---

## 8. 单 root 动作选择

```bash
python scripts/run_select.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --root-id light2d_000220 \
  --split test \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --num-samples 16 \
  --beta 0.2 \
  --device auto \
  --torch-threads 1
```

选择流程严格限制在同一 root 的最小 harm bin 内。

---

## 9. Baselines 与 residual-action 诊断

Direct-action baseline：

```bash
python scripts/train_baseline.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --out-dir runs/baselines \
  --baseline direct_action_to_risk \
  --epochs 50 \
  --device auto \
  --torch-threads 1
```

Residual-action diagnostic：

```bash
python scripts/diagnose_residual_action.py \
  --data data/metadrive_mrvp_clean.jsonl \
  --feature-source predicted_msrt \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --target recoverable \
  --admissible-only \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_predicted.json \
  --device auto \
  --torch-threads 1
```

Feature sources：

```text
true_clean
predicted_msrt
leaky_current
xplus_only
world_only
tokens_only
```

输出核心字段：

```json
{
  "probe0_auc": 0.0,
  "probe1_auc": 0.0,
  "delta_auc": 0.0,
  "probe0_balanced_acc": 0.0,
  "probe1_balanced_acc": 0.0,
  "delta_balanced_acc": 0.0,
  "probe0_nll": 0.0,
  "probe1_nll": 0.0,
  "delta_nll": 0.0
}
```

建议判断标准：

```text
Delta_AUC <= 0.01：强支持
0.01 < Delta_AUC <= 0.03：可接受
0.03 < Delta_AUC <= 0.05：有风险
Delta_AUC > 0.05：action shortcut 仍明显
```

---

## 10. Tests

```bash
pytest -q
```

测试覆盖：

```text
schema no audit fallback
world_plus no target/recovery leakage
RPN strategy_traj from dynamics rollout
RPN no z_mech fallback
selection harm gate
smoke forward/trainability
```

---

## 11. 当前实现重点

- `event_tokens` 缺失时输出全零和 `has_event_tokens=0`，不会使用 `audit_mech/z_mech` fallback。
- `world_plus` 已从 clean post-event observable/predictable features 构造。
- MSRT transition decoder 显式 event-conditioned。
- RPN 轨迹由 degraded dynamics rollout 产生，不再由 MLP 自由预测。
- RPN margins 基于 `strategy_traj + controls + world_plus + deg` 验证。
- RPN train/calibrate/evaluate 的 clean oracle 路径使用 MSRT-generated tokens。
- leaky/debug 路径必须显式开启，并命名为 `Oracle_leaky_do_not_report`。
