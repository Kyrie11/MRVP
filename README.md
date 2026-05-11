# MRVP PyTorch 实现（新版论文接口）

本仓库已按新版论文重构为 **MSRT event-tokenizer + RPN strategy-branch viability** 的实现。旧版代码中把 `z_mech` 当作核心机制向量直接喂给 RPN；新版已改为：

```text
pre-impact scene + candidate action
  -> MSRT: Action-conditioned Recovery Event Tokenizer
     输出 event_type/event_time, x_minus/x_plus, event_tokens, world_plus, deg
  -> RPN: Policy-conditioned Recovery Viability Network
     基于 event_tokens + world_plus + x_plus + deg 解码多条 recovery strategies
     输出五个 strategy-conditioned signed margins
  -> calibration + CVaR action selection
```

`z_mech` / `audit_mech` 仍被保留，但只用于审计、probe 和兼容旧数据，不再作为新版 RPN 的主输入。

---

## 1. 目录结构

根目录只保留包、配置、文档和测试；所有可执行脚本已移到 `scripts/`。

```text
MRVP-main/
  README.md
  requirements.txt
  pyproject.toml
  configs/default.yaml
  docs/paper_method_mapping.md

  scripts/
    generate_metadrive.py          # MetaDrive/light2d counterfactual 数据生成
    generate_synthetic.py          # smoke-test 合成数据
    train.py                       # MSRT/RPN/pair fine-tuning
    calibrate.py                   # group/scenario calibration
    evaluate.py                    # oracle true-transition 上界评估
    evaluate_predicted_mrvp.py     # MSRT samples -> RPN -> calibration -> CVaR
    run_select.py                  # 单 root 在线式动作选择
    diagnose_residual_action.py    # residual action sufficiency 诊断
    train_baseline.py              # direct-action / scalar / unstructured baselines
    scenario_builder.py            # CARLA-MRVP 入口，仍为后续扩展
    preprocess_public.py
    pretrain_context.py

  mrvp/
    data/
      schema.py                    # 新版 JSONL schema 与兼容旧字段的 loader
      dataset.py                   # MRVPDataset / root-level loader
      metadrive_rollout.py         # MetaDrive/light2d rollout generator
      synthetic.py
    models/
      msrt.py                      # Action-conditioned Recovery Event Tokenizer
      rpn.py                       # Policy-conditioned Recovery Viability Network
      baselines.py
    calibration.py
    evaluation.py
    selection.py
    carla/                         # CARLA adapter，后续继续扩展
```

---

## 2. 安装

```bash
cd MRVP-main
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

脚本已自动把项目根目录加入 `sys.path`，通常不需要手动设置 `PYTHONPATH`。CPU 上建议固定线程数：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

可选安装 MetaDrive：

```bash
pip install metadrive-simulator
```

未安装 MetaDrive 时，`scripts/generate_metadrive.py --backend auto` 会自动回退到内置 `light2d` rollout，保证同一 schema 能跑通。CARLA 仍作为后续扩展保留在 `mrvp/carla/`。

---

## 3. 新版数据 schema

每一行 JSONL 表示同一个 root scenario 下的一个 candidate emergency action。`MRVPDataset` 可以直接加载你已经生成的 `metadrive_mrvp.jsonl`。

### 3.1 推荐字段

| 字段 | 含义 |
| --- | --- |
| `root_id` | 同一初始状态、地图、交通、摩擦和触发条件的 counterfactual 组 |
| `split` | root-level split：`train/val/cal/test` |
| `action_id`, `action_name`, `action_vec` | 候选紧急动作；默认 8 个 action |
| `o_hist` | `[T,A,F]` ego 与周围 actor 历史 |
| `h_ctx` | route、road、boundary、friction、density 等上下文 |
| `x_t` | 执行候选 action 前的状态 |
| `rho_imp`, `harm_bin` | first-impact harm surrogate 及 harm gate bin |
| `event_type` | `none/contact/boundary/stability/control` |
| `event_time` | 事件发生或 transition 完成时间 |
| `x_minus`, `x_plus` | recovery-critical transition 前后状态 |
| `deg` | steering/brake/throttle authority、delay、friction、damage class |
| `world_plus` | post-event local recovery world，loader 会 flatten 成固定向量 |
| `teacher_u` | recovery teacher controls，shape `[H,3]` |
| `teacher_traj` | recovery teacher states，shape `[H+1,12]` |
| `m_star` | 五个 bottleneck signed margins：`sec/road/stab/ctrl/return` |
| `b_star`, `s_star` | active bottleneck 与最差 margin |
| `audit_mech` | 可读审计字段，不作为新版 RPN 主输入 |
| `calib_group` | event、side、friction、damage、density、town/family 等校准分组 |

状态维度为 12：

```text
[p_x, p_y, psi, v_x, v_y, yaw_rate, a_x, a_y, beta, delta, F_b, F_x]
```

控制维度为 3：

```text
[delta, F_b, F_x]
```

### 3.2 旧字段兼容

旧版字段仍可加载：

| 旧字段 | 新字段别名 |
| --- | --- |
| `d_deg` | `deg` |
| `z_mech` | `audit_mech` / fallback event tokens |
| `r_star` | `m_star` |

因此旧数据不会直接失效，但新版训练建议使用包含 `event_type/event_time/world_plus/teacher_u/teacher_traj/m_star` 的 `metadrive_mrvp.jsonl`。

---

## 4. 生成或加载 MetaDrive 数据

你已经生成了 `metadrive_mrvp.jsonl`，可以直接放在 `data/` 下：

```bash
mkdir -p data
# 例如：data/metadrive_mrvp.jsonl
```

重新生成数据：

```bash
python scripts/generate_metadrive.py \
  --output data/metadrive_mrvp.jsonl \
  --n-roots 1000 \
  --backend auto \
  --seed 13
```

只用内置轻量仿真做快速验证：

```bash
python scripts/generate_metadrive.py \
  --output data/light2d_mrvp.jsonl \
  --n-roots 1000 \
  --backend light2d \
  --seed 13
```

生成器流程：

```text
root scene
  -> 对 8 个 candidate actions 分别 reset 同一场景
  -> open-loop emergency action rollout
  -> 提取 event_type, x_minus, x_plus, deg, audit_mech
  -> degraded recovery controller rollout
  -> 从 recovery trajectory 计算 m_star 与 teacher_u/teacher_traj
  -> 写入 JSONL
```

默认 test roots 会更难一些（更低 friction、更高 actor density/损伤），用于 shift split。需要同分布 sanity check 时加入：

```bash
--no-shift-test
```

---

## 5. Smoke test

没有 simulator 也可以先验证代码是否完整：

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
  --device auto \
  --torch-threads 1
```

测试：

```bash
pytest -q
```

---

## 6. 训练新版 MSRT/RPN

### 6.1 一条命令跑完整流程

```bash
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage all \
  --epochs 30 \
  --batch-size 256 \
  --device auto \
  --torch-threads 1
```

这会依次保存：

```text
runs/metadrive_mrvp/msrt.pt
runs/metadrive_mrvp/rpn.pt
runs/metadrive_mrvp/rpn_finetuned.pt
```

### 6.2 分阶段训练

MSRT：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage msrt \
  --epochs 30 \
  --batch-size 256 \
  --mixture-count 5 \
  --lambda-event 1.0 \
  --lambda-token 0.5 \
  --lambda-world 0.5 \
  --lambda-deg 1.0 \
  --lambda-probe 0.1 \
  --device auto \
  --torch-threads 1
```

RPN：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage rpn \
  --epochs 50 \
  --batch-size 256 \
  --strategy-count 6 \
  --recovery-horizon 30 \
  --lambda-strat 0.5 \
  --lambda-dyn 0.05 \
  --lambda-ctrl 0.05 \
  --device auto \
  --torch-threads 1
```

Same-root pair fine-tuning：

```bash
python scripts/train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage finetune \
  --rpn-init runs/metadrive_mrvp/rpn.pt \
  --epochs 15 \
  --batch-size 128 \
  --eps-s 0.25 \
  --pair-margin 0.05 \
  --lambda-ord 1.0 \
  --device auto \
  --torch-threads 1
```

默认 compact 实现使用 `token-count=16`、`token-dim=32`。如果要完全贴近论文大模型设置，可以在资源允许时增大 `--token-dim` 和 `--hidden-dim`，但需要保证训练、校准、评估、选择阶段参数一致。

---

## 7. Calibration

```bash
python scripts/calibrate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --output runs/metadrive_mrvp/calibration.json \
  --split cal \
  --delta-b 0.02 \
  --n-min 20 \
  --device auto \
  --torch-threads 1
```

Calibration 以 root scenario 为单位，按 `calib_group` 的 `full -> medium -> coarse -> global` 层级回退，避免小组样本不足。

---

## 8. 评估

### 8.1 Oracle true-transition 上界

```bash
python scripts/evaluate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --split test \
  --output runs/metadrive_mrvp/eval_oracle_test.json \
  --device auto \
  --torch-threads 1
```

`evaluate.py` 使用数据集中真实 `x_plus/deg/event_tokens/world_plus`，用于估计 RPN 上界；它不是完整推理链路。

### 8.2 Predicted-MRVP 真推理链路

```bash
python scripts/evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
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

该脚本会对同一个 root 下所有 candidate actions 一起打分，并输出：

```text
Oracle_MRVP_true_transition
Oracle_MRVP_uncalibrated
Predicted_MRVP_MSRT_CVaR
Predicted_MRVP_MSRT_mean_risk
Severity_only
Weighted_post_impact_cost
Teacher_oracle
```

同时生成：

```text
runs/metadrive_mrvp/eval_predicted_test.json
runs/metadrive_mrvp/eval_predicted_test.csv
runs/metadrive_mrvp/eval_predicted_test_per_root.jsonl
```

---

## 9. 单 root 动作选择

```bash
python scripts/run_select.py \
  --data data/metadrive_mrvp.jsonl \
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

选择流程：

```text
1. 在同一 root 内找最小 harm_bin 的 admissible action set
2. 对每个 admissible action 采样 M 个 MSRT post-event recovery worlds
3. RPN 对每个 sample 预测五个 bottleneck margins
4. 减 calibration quantile，得到 conservative lower bounds
5. 对 violation depth [-min_b margin_b]_+ 计算 empirical CVaR
6. 选择 CVaR 最小的 action
```

---

## 10. Baselines 与 residual-action 诊断

Direct-action baseline：

```bash
python scripts/train_baseline.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline direct_action_to_risk \
  --epochs 50 \
  --device auto \
  --torch-threads 1
```

把 baseline 加入 Predicted-MRVP 评估：

```bash
python scripts/evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --direct-model runs/baselines/direct_action_to_risk.pt \
  --direct-model-type direct_action_to_risk \
  --split test \
  --num-samples 16 \
  --beta 0.2 \
  --output runs/metadrive_mrvp/eval_predicted_with_direct.json \
  --device auto \
  --torch-threads 1
```

Residual-action diagnostic：

```bash
python scripts/diagnose_residual_action.py \
  --data data/metadrive_mrvp.jsonl \
  --feature-source true \
  --target recoverable \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_true.json \
  --device auto \
  --torch-threads 1
```

用 MSRT 预测表示诊断：

```bash
python scripts/diagnose_residual_action.py \
  --data data/metadrive_mrvp.jsonl \
  --feature-source predicted_msrt \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --target recoverable \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_predicted.json \
  --device auto \
  --torch-threads 1
```

Probe0 输入：

```text
x_plus, deg, event_tokens, world_plus, h_ctx
```

Probe1 额外输入：

```text
action_vec, action_id_onehot
```

如果 Probe1 相对 Probe0 的提升很大，说明 event tokens / world_plus 仍有 residual action shortcut 或信息缺失。

---

## 11. CARLA 后续扩展

当前主要路径基于 MetaDrive/light2d。CARLA 入口仍保留：

```bash
python scripts/scenario_builder.py \
  --host 127.0.0.1 \
  --port 2000 \
  --tm-port 8000 \
  --output data/carla_mrvp.jsonl \
  --n-roots 1000 \
  --seed 13 \
  --towns Town03,Town04,Town05,Town10HD \
  --fixed-delta-seconds 0.05
```

后续 CARLA adapter 应输出同一套字段：`event_type/event_time/x_plus/deg/world_plus/teacher_u/teacher_traj/m_star/audit_mech`，这样训练和评估脚本无需再改。

---

## 12. 当前实现重点

- `MRVPDataset` 已支持加载 `metadrive_mrvp.jsonl`，并兼容旧字段。
- MSRT 已改为预测 `event_tokens` 与 `world_plus`，`z_mech` 只作审计/probe。
- RPN 已改为多 strategy branches：`strategy_u`、`strategy_traj`、`branch_margins`，最终取 best branch 的 soft-min viability。
- MetaDrive/light2d generator 已写入新版 preferred schema，并保留旧别名方便旧分析脚本迁移。
- 根目录脚本已移动到 `scripts/`，删除了不必要的根目录 re-export 和缓存文件。
