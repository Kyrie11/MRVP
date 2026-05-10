# MRVP PyTorch 落地实现

本工程按照论文 `Mechanism-Sufficient Recovery Viability Planning for Near-Unavoidable Collisions` 的 **Method / Appendix / Experiments** 三部分组织代码。核心流程是：先用 first-impact harm bin 做硬门控，只在最小伤害等价类内选择动作；再用 MSRT 从 pre-impact scene + action 预测动作诱导的 post-transition recovery problem；最后由 RPN 预测五个恢复瓶颈 signed margins，经 group/scenario-level calibration 后按 CVaR tail risk 选动作。

本版本补齐了用于论证核心 claim 的三条链路：

1. **Oracle-MRVP vs Predicted-MRVP**：`evaluate.py` 只评估真实 `x_plus/d_deg/z_mech` 输入下的 oracle 上界；新增 `evaluate_predicted_mrvp.py` 会真正调用 `MSRT.sample -> RPN -> calibration -> CVaR`，验证 MSRT 预测出的机制变量是否能支撑动作选择。
2. **Residual-action sufficiency diagnostic**：新增 `diagnose_residual_action.py`，比较 Probe0 `(x_plus,d,z,h)->recoverability` 与 Probe1 `(x_plus,d,z,h,a)->recoverability`。若 Probe1 相比 Probe0 的 AUC / balanced accuracy / NLL 增益很小，才支持 mechanism-sufficient；若增益大，说明 action shortcut 或机制变量遗漏仍然存在。
3. **Trajectory-labeled MetaDrive 数据构造**：新增 `generate_metadrive.py` 与 `mrvp/data/metadrive_rollout.py`。它按同一 root scenario 展开多个 candidate actions，执行 open-loop emergency rollout，再执行 degraded recovery controller rollout，并从 recovery trajectory 计算 `r_star`，避免把 recoverability claim 直接写进静态公式。

---

## 1. 工程结构

```text
mrvp_pytorch/
  README.md
  requirements.txt
  configs/default.yaml

  # 数据生成
  generate_synthetic.py          # 原 smoke-test 合成数据，仍保留，仅用于工程自检
  generate_metadrive.py          # 新增：MetaDrive / light2d trajectory-labeled MRVP 数据
  scenario_builder.py            # CARLA-MRVP 数据生成入口

  # 论文附录组件
  transition_extractor.py
  mechanism_labels.py
  recovery_teachers.py
  targets.py

  # 训练与校准
  train.py                       # MSRT、RPN、pair fine-tuning；新增 --lambda-suf
  train_baseline.py              # direct-action / unstructured latent / scalar baseline
  calibrate.py                   # group/scenario conformal lower bounds

  # 推理与 claim 评估
  select.py                      # 单 root 在线式动作选择
  evaluate.py                    # Oracle-MRVP：真实 transition/mechanism 输入下的上界评估
  evaluate_predicted_mrvp.py     # 新增：Predicted-MRVP 真推理路径评估
  diagnose_residual_action.py    # 新增：mechanism sufficiency / action shortcut 诊断

  mrvp/
    data/
      dataset.py                 # row-level dataset；新增 iter_root_batches root-level loader
      pairs.py                   # same-root severity-equivalent pair mining
      schema.py                  # JSONL schema / vector layout
      synthetic.py
      metadrive_rollout.py       # 新增：MetaDrive/light2d rollout 数据构造
    models/
      msrt.py                    # 新增可选 gradient-reversal action adversary
      rpn.py
      baselines.py
    calibration.py
    evaluation.py                # 新增 evaluate_selected_indices，用于 root-level sampled selector
    selection.py
```

`docs/paper_method_mapping.md` 给出论文每个公式/附录模块到代码文件的映射。

---

## 2. 环境安装

推荐 Python 3.10。仅跑 PyTorch、synthetic、light2d 数据不需要安装 CARLA 或 MetaDrive。

```bash
cd mrvp_pytorch
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

CPU 上建议固定线程数，否则小 batch 的 GRU/MLP 可能被 BLAS 线程调度拖慢：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

可选依赖按需安装：

```bash
# MetaDrive 数据构造；不安装时 generate_metadrive.py --backend auto 会回退到 light2d
pip install metadrive-simulator

# Argoverse 2 parquet
pip install pyarrow

# nuScenes
pip install nuscenes-devkit

# CommonRoad
pip install commonroad-io

# Waymo Open Motion，版本需和本地 TensorFlow 匹配
pip install tensorflow waymo-open-dataset-tf-2-12-0
```

CARLA 不建议随意 `pip install carla`，应使用和 simulator 完全匹配的 CARLA egg，例如：

```bash
export CARLA_ROOT=/opt/carla-0.9.15
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg:$CARLA_ROOT/PythonAPI/carla
```

若没有以 editable 方式安装包，运行脚本前可设置：

```bash
export PYTHONPATH=.
```

---

## 3. 数据集 schema 与加载方式

每一行 JSONL 表示一个 root scenario 下的一个 candidate action。核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `root_id` | 同一初始状态、map/road、traffic seed、friction、trigger 下的 counterfactual 组 |
| `split` | root-level split：`train/val/cal/test`；不要按 action 随机切分 |
| `action_id`, `action_name`, `action_vec` | 候选紧急动作 |
| `o_hist` | `[T,A,F]` ego 与 surrounding actors 历史 |
| `h_ctx` | route corridor、road width、boundary side、friction、hazard density 等 context |
| `rho_imp`, `harm_bin` | first-impact harm surrogate 及 monotone harm bin |
| `x_minus`, `x_plus` | emergency transition 前后状态 |
| `d_deg` | residual steering/brake/throttle authority、delay、friction、damage class |
| `z_mech` | 32 维 mechanism descriptor |
| `r_star` | 五个 teacher bottleneck signed margins：secondary、road、stability、control、return |
| `b_star`, `s_star` | active bottleneck 与 aggregate worst margin |
| `calib_group` | contact type、side、friction bin、damage class、density、town/family 等 calibration group |

训练阶段可以继续使用 row-level `MRVPDataset + DataLoader`。动作选择和 claim 评估必须用 root-level loader，因为同一 root 的所有 counterfactual actions 必须一起打分：

```python
from mrvp.data.dataset import MRVPDataset, iter_root_batches

ds = MRVPDataset("data/metadrive_mrvp.jsonl", split="test")
for root_id, indices, rows, batch in iter_root_batches(ds):
    # rows/batch 包含同一 root 下所有 candidate actions
    ...
```

---

## 4. 先跑一个无 simulator 的 smoke test

这一步只验证 dataset、MSRT、RPN、pair mining、calibration、evaluation、selection 是否能跑通。原 synthetic 数据仍然是公式化 teacher，不能作为 claim 的主要证据。

```bash
python generate_synthetic.py \
  --output data/synthetic_mrvp.jsonl \
  --n-roots 240 \
  --seed 7

python train.py \
  --data data/synthetic_mrvp.jsonl \
  --out-dir runs/synthetic \
  --stage all \
  --epochs 5 \
  --batch-size 128 \
  --device auto \
  --torch-threads 1

python calibrate.py \
  --data data/synthetic_mrvp.jsonl \
  --rpn runs/synthetic/rpn_finetuned.pt \
  --output runs/synthetic/calibration.json \
  --split cal \
  --torch-threads 1

python evaluate.py \
  --data data/synthetic_mrvp.jsonl \
  --rpn runs/synthetic/rpn_finetuned.pt \
  --calibration runs/synthetic/calibration.json \
  --output runs/synthetic/eval_oracle_test.json \
  --split test \
  --torch-threads 1
```

`evaluate.py` 输出的 `Oracle_MRVP_true_transition` 是上界诊断：它使用数据集中真实 `x_plus/d_deg/z_mech`，不能证明 MSRT 在真实推理时有用。

---

## 5. MetaDrive / light2d trajectory-labeled 数据构造

### 5.1 推荐命令

若本地装有 MetaDrive：

```bash
python generate_metadrive.py \
  --output data/metadrive_mrvp.jsonl \
  --n-roots 1000 \
  --backend metadrive \
  --seed 13
```

若没有 MetaDrive，先用内置 light2d rollout 做同 schema 的 proof-of-concept：

```bash
python generate_metadrive.py \
  --output data/light2d_mrvp.jsonl \
  --n-roots 1000 \
  --backend light2d \
  --seed 13
```

`--backend auto` 会优先尝试 MetaDrive，失败后回退到 light2d：

```bash
python generate_metadrive.py \
  --output data/mrvp_rollout.jsonl \
  --n-roots 1000 \
  --backend auto \
  --seed 13
```

默认会让 test roots 具有更难的 friction / actor density / damage 分布，用于 shift split。若只想同分布 sanity check：

```bash
python generate_metadrive.py \
  --output data/mrvp_rollout_iid.jsonl \
  --n-roots 1000 \
  --backend light2d \
  --no-shift-test
```

### 5.2 数据生成逻辑

每个 root scenario 会扩展为 8 个 candidate actions：

```text
hard_brake, brake_left, brake_right, maintain,
mild_left, mild_right, boundary_side_steer, corridor_side_steer
```

每个 action 的 label 由轨迹产生，而不是直接由 action 公式生成：

```text
root scene
  -> reset 同一 road / friction / actor setting
  -> open-loop emergency action rollout 0.7s
  -> extract x_minus, x_plus, contact/boundary mechanism, d_deg, z_mech
  -> degraded recovery controller rollout 4s
  -> compute r_star from recovery trajectory
```

当前 generator 覆盖的场景族包括：

```text
rear_end_blocked_forward_corridor
side_swipe_near_boundary
oblique_intersection_impact
cut_in_unavoidable_contact
boundary_critical_non_contact
low_friction_recovery
actuator_degradation_after_impact
dense_agent_secondary_exposure
```

light2d backend 是 deterministic bicycle-like dynamics + moving actors + degraded recovery controller，用于快速验证 claim 链路；MetaDrive backend 使用 MetaDrive 环境执行相同 counterfactual action/recovery 协议，并将 native crash/off-road signals 映射回 MRVP schema。

---

## 6. 训练模型

### 6.1 MSRT

```bash
python train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage msrt \
  --epochs 30 \
  --batch-size 256 \
  --lr 3e-4 \
  --mixture-count 5 \
  --lambda-mech 1.0 \
  --lambda-phys 0.2 \
  --lambda-suf 0.0 \
  --device auto \
  --torch-threads 1
```

`--lambda-suf` 是新增的可选 gradient-reversal action adversary：它让一个 action classifier 从 `z_mech` 预测 `action_id`，并通过 gradient reversal 让 `z_mech` 减少 raw action identity 泄漏。默认保持 `0.0`。建议先跑 `diagnose_residual_action.py`，发现 residual action gain 很大时再尝试小权重，例如 `0.02` 或 `0.05`。

### 6.2 RPN

```bash
python train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage rpn \
  --epochs 50 \
  --batch-size 256 \
  --lr 3e-4 \
  --lambda-act 0.5 \
  --lambda-bd 2.0 \
  --sigma-bd 0.5 \
  --device auto \
  --torch-threads 1
```

### 6.3 Same-root severity-equivalent pair fine-tuning

```bash
python train.py \
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

也可以直接：

```bash
python train.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/metadrive_mrvp \
  --stage all \
  --epochs 30 \
  --device auto \
  --torch-threads 1
```

正式实验建议分阶段训练，便于检查 transition NLL、mechanism loss、RPN profile loss 和 pair ordering loss。

---

## 7. Calibration

```bash
python calibrate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --output runs/metadrive_mrvp/calibration.json \
  --split cal \
  --delta-b 0.02 \
  --n-min 20 \
  --device auto \
  --torch-threads 1
```

实现细节对应 Appendix：对每个 calibration root scenario、group、bottleneck，先在 admissible action set 上取 worst optimistic residual，再计算 group quantile；小 group 会按 `full -> medium -> coarse -> global` 回退。

---

## 8. Claim 评估主流程

### 8.1 Oracle-MRVP 上界

```bash
python evaluate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --split test \
  --output runs/metadrive_mrvp/eval_oracle_test.json \
  --device auto \
  --torch-threads 1
```

输出方法名：

- `Oracle_MRVP_true_transition`：RPN 使用真实 `x_plus/d_deg/z_mech`，是结构化 RPN 的上界。
- `Uncalibrated_Oracle_MRVP`：不做 calibration 的 oracle。
- `Scalar_recoverability_network_proxy`：把 oracle profile 均值广播成 scalar 的弱 proxy；正式 scalar baseline 应使用 `train_baseline.py --baseline scalar_recoverability` 训练。
- `Severity_only`、`Weighted_post_impact_cost`、`Teacher_oracle`。

### 8.2 Predicted-MRVP 真推理路径

```bash
python evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --split test \
  --num-samples 64 \
  --beta 0.9 \
  --output runs/metadrive_mrvp/eval_predicted_test.json \
  --device auto \
  --torch-threads 1
```

该脚本会输出：

- `Oracle_MRVP_true_transition`：真实机制变量上界。
- `Predicted_MRVP_MSRT_CVaR`：真实论文推理路径，使用 MSRT 多样本 + RPN + calibration + CVaR。
- `Predicted_MRVP_MSRT_mean_risk`：同样使用 MSRT 多样本，但用 mean violation risk 替代 CVaR，是 CVaR ablation。
- `Severity_only`、`Weighted_post_impact_cost`、`Teacher_oracle`。

同时会生成：

```text
runs/metadrive_mrvp/eval_predicted_test.json
runs/metadrive_mrvp/eval_predicted_test.csv
runs/metadrive_mrvp/eval_predicted_test_per_root.jsonl
```

`*_per_root.jsonl` 记录每个 root 下每个 action 的 `risk_cvar / risk_mean / p_violation / mean_lower_bounds`，用于分析 CVaR tail samples 是否改变了动作排序。

### 8.3 Direct-action baseline

训练 baseline：

```bash
python train_baseline.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline direct_action_to_risk \
  --epochs 50 \
  --device auto \
  --torch-threads 1

python calibrate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/baselines/direct_action_to_risk.pt \
  --model-type direct_action_to_risk \
  --output runs/baselines/direct_calibration.json \
  --split cal \
  --device auto \
  --torch-threads 1
```

在 Predicted-MRVP 评估中一起报告：

```bash
python evaluate_predicted_mrvp.py \
  --data data/metadrive_mrvp.jsonl \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --direct-model runs/baselines/direct_action_to_risk.pt \
  --direct-calibration runs/baselines/direct_calibration.json \
  --split test \
  --num-samples 64 \
  --beta 0.9 \
  --output runs/metadrive_mrvp/eval_predicted_with_direct.json \
  --device auto \
  --torch-threads 1
```

真正支持论文 claim 的初步证据应当是：

```text
Predicted_MRVP_MSRT_CVaR 优于 Direct_baseline_direct_action_to_risk，
并且接近 Oracle_MRVP_true_transition；
在 shift test split 上 shift_regret / violation_depth / recovery_success 仍保持优势。
```

### 8.4 Scalar recoverability baseline

```bash
python train_baseline.py \
  --data data/metadrive_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline scalar_recoverability \
  --epochs 50 \
  --device auto \
  --torch-threads 1

python evaluate.py \
  --data data/metadrive_mrvp.jsonl \
  --rpn runs/baselines/scalar_recoverability.pt \
  --model-type direct_action_to_risk \
  --scalar-rpn \
  --split test \
  --output runs/baselines/scalar_eval.json \
  --device auto \
  --torch-threads 1
```

这比旧的 `Scalar_recoverability_network_proxy` 公平，因为它是单独训练出的 scalar network。

---

## 9. Residual-action sufficiency diagnostic

用真实机制变量做诊断：

```bash
python diagnose_residual_action.py \
  --data data/metadrive_mrvp.jsonl \
  --feature-source true \
  --target recoverable \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_true.json \
  --device auto \
  --torch-threads 1
```

用 MSRT 预测出的机制变量做诊断：

```bash
python diagnose_residual_action.py \
  --data data/metadrive_mrvp.jsonl \
  --feature-source predicted_msrt \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --target recoverable \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_predicted.json \
  --device auto \
  --torch-threads 1
```

输出解释：

| 指标 | 解释 |
| --- | --- |
| `probe0_no_action` | 输入 `(x_plus,d_deg,z_mech,h_ctx)` |
| `probe1_with_action` | 输入 `(x_plus,d_deg,z_mech,h_ctx,action_vec,action_id_onehot)` |
| `delta_auc_probe1_minus_probe0` | Probe1 AUC 增益；越接近 0 越支持 sufficiency |
| `delta_balanced_acc_probe1_minus_probe0` | Probe1 balanced accuracy 增益 |
| `delta_nll_probe1_minus_probe0` | Probe1 NLL - Probe0 NLL；负值表示 action 额外降低 NLL |

经验解释规则：

```text
Delta_action < 1-2%：机制变量基本充分，raw action identity 没有明显额外信息。
Delta_action > 5%：action shortcut 明显，z 或 x_plus/d_deg/h_ctx 可能漏掉了 action-induced recovery 信息。
```

也可以诊断 active bottleneck：

```bash
python diagnose_residual_action.py \
  --data data/metadrive_mrvp.jsonl \
  --feature-source true \
  --target bottleneck \
  --epochs 20 \
  --output runs/metadrive_mrvp/residual_action_bottleneck.json
```

---

## 10. 单 root 动作选择

```bash
python run_select.py \
  --data data/metadrive_mrvp.jsonl \
  --root-id light2d_000220 \
  --split test \
  --msrt runs/metadrive_mrvp/msrt.pt \
  --rpn runs/metadrive_mrvp/rpn_finetuned.pt \
  --calibration runs/metadrive_mrvp/calibration.json \
  --num-samples 64 \
  --beta 0.9 \
  --device auto \
  --torch-threads 1
```

推理顺序：

1. `admissible_indices` 找 root 内最小 `harm_bin`。
2. 对每个 admissible action 采样 `M` 个 MSRT transition。
3. RPN 预测每个 sample 的五个 margins。
4. 根据 action 的 calibration group 减去 quantile，得到 conservative lower bounds。
5. 取 worst bottleneck `min_b lower_r_b`。
6. 对 violation depth `[-V]_+` 计算 empirical CVaR。
7. 选择 CVaR 最小的 action。

若要做 expected-risk ablation，把 `--beta 0.0` 传给 `select.py`；此时 tail average 退化为样本平均 violation depth。

---

## 11. CARLA-MRVP 数据集构造

MetaDrive/light2d 用于快速 proof-of-concept。若要运行 CARLA generator，先启动 CARLA server：

```bash
$CARLA_ROOT/CarlaUE4.sh \
  -quality-level=Low \
  -RenderOffScreen \
  -carla-rpc-port=2000
```

生成 CARLA-MRVP JSONL：

```bash
python scenario_builder.py \
  --host 127.0.0.1 \
  --port 2000 \
  --tm-port 8000 \
  --output data/carla_mrvp.jsonl \
  --n-roots 1000 \
  --seed 13 \
  --towns Town03,Town04,Town05,Town10HD \
  --fixed-delta-seconds 0.05
```

CARLA generator 会把 world 和 Traffic Manager 设成 synchronous mode，并设置 fixed delta seconds，使同一个 root scenario 的多个 candidate action 在相同 seed / map / traffic / trigger 下 rollout，避免 counterfactual 泄漏。

---

## 12. 公开数据集 context / scene encoder 预训练

论文指出 nuPlan、Waymo Open Motion、Argoverse 2、INTERACTION、CommonRoad、highD、nuScenes 可用于 context、map-affordance、interaction pretraining，但这些数据通常没有完整 MRVP counterfactual labels。因此本工程把它们统一转成：

```json
{
  "dataset": "highD",
  "scenario_id": "...",
  "o_hist": [[[...]]],
  "h_ctx": [...],
  "future_delta": [dx, dy],
  "affordance": [drivable_width, hazard_distance, return_corridor_length, actor_density]
}
```

示例：

```bash
python preprocess_public.py \
  --dataset highd \
  --input /data/highD \
  --output data/pretrain_highd.jsonl \
  --max-records 200000

python preprocess_public.py \
  --dataset argoverse2 \
  --input /data/argoverse2/motion-forecasting \
  --output data/pretrain_av2.jsonl

cat data/pretrain_highd.jsonl data/pretrain_av2.jsonl > data/pretrain_all.jsonl

python pretrain_context.py \
  --data data/pretrain_all.jsonl \
  --output runs/pretrain/context_encoder.pt \
  --epochs 10 \
  --batch-size 256 \
  --device auto \
  --torch-threads 1
```

当前 `train.py` 会从头训练 MSRT/RPN 的 context encoder。若要使用预训练权重，可在自己的实验脚本中把 `runs/pretrain/context_encoder.pt` 的 state dict load 到 `MSRT.context_encoder` 和 `RecoveryProfileNetwork.context_encoder`。

---

## 13. 推荐结果表

建议论文初步实验至少报告以下方法：

| 方法 | 对应命令 |
| --- | --- |
| `Severity_only` | `evaluate_predicted_mrvp.py` 自动输出 |
| `Direct_baseline_direct_action_to_risk` | 先 `train_baseline.py --baseline direct_action_to_risk`，再在 `evaluate_predicted_mrvp.py` 传 `--direct-model` |
| `Oracle_MRVP_true_transition` | `evaluate.py` 或 `evaluate_predicted_mrvp.py` 自动输出 |
| `Predicted_MRVP_MSRT_mean_risk` | `evaluate_predicted_mrvp.py` 自动输出 |
| `Predicted_MRVP_MSRT_CVaR` | `evaluate_predicted_mrvp.py` 自动输出 |
| `Scalar_recoverability` | `train_baseline.py --baseline scalar_recoverability` + `evaluate.py --scalar-rpn` |
| `Residual-action Probe0/Probe1` | `diagnose_residual_action.py` |

核心指标：

```text
envelope_violation              # harm gate 是否被破坏，理论上应为 0
pair_accuracy                   # same-root severity-equivalent pair 排序
frr                             # false recoverable rate
worst_bottleneck_violation_depth
closed_loop_recovery_success_proxy
active_bottleneck_f1
coverage / coverage_sec / ...
shift_regret
```

---

## 14. 重要实现说明

1. **`evaluate.py` 是 oracle 上界，不再把它称作真实 MRVP claim。** 真实 claim 评估请用 `evaluate_predicted_mrvp.py`。
2. **public datasets 只做预训练，不产生完整 MRVP labels。** 它们通常不同时提供 counterfactual emergency actions、post-impact reset、degradation 和 recovery-controller bottleneck margins。
3. **trajectory-labeled generator 不等于高保真碰撞物理。** light2d 是快速 proof-of-concept；MetaDrive 的碰撞/损伤也弱于 CARLA。但二者都比旧 synthetic 更适合验证“同等 first-impact harm 下 post-transition recoverability 不同”。
4. **Calibration 必须按 root scenario split。** `calibrate.py` 用 root-level worst optimistic residual，避免同一 root 的 action 相关性破坏 coverage。
5. **Mechanism sufficiency 不能只靠 loss 声称。** 必须报告 `diagnose_residual_action.py` 的 Probe0/Probe1 residual gain。
6. **CVaR 必须用 MSRT 多样本评估。** 离线单 profile/action 的 CVaR 会退化成普通 violation depth；`evaluate_predicted_mrvp.py` 和 `select.py` 才是正确的 sampling path。

---

## 15. 参考格式来源

- MetaDrive Documentation: https://metadrive-simulator.readthedocs.io/en/latest/index.html
- MetaDrive GitHub: https://github.com/metadriverse/metadrive
- CARLA Python API: https://carla.readthedocs.io/en/latest/python_api/
- CARLA Traffic Manager synchronous mode: https://carla.readthedocs.io/en/latest/adv_traffic_manager/
- CARLA fixed time-step: https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/
- Waymo Open Motion Scenario proto: https://waymo.com/open/data/motion/
- Argoverse 2 Motion Forecasting user guide: https://argoverse.github.io/user-guide/datasets/motion_forecasting.html
- nuPlan devkit: https://github.com/motional/nuplan-devkit
- CommonRoad-io docs: https://cps.pages.gitlab.lrz.de/commonroad/commonroad-io/
- highD dataset format: https://levelxdata.com/wp-content/uploads/2023/10/highD-Format.pdf
- INTERACTION dataset scripts: https://github.com/interaction-dataset/interaction-dataset
- nuScenes tutorial: https://www.nuscenes.org/tutorials/nuscenes_tutorial.html
