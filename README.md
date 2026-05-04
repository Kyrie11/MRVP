# MRVP PyTorch 落地实现

本工程按照论文 `Mechanism-Sufficient Recovery Viability Planning for Near-Unavoidable Collisions` 的 **Method / Appendix / Experiments** 三部分组织代码。核心思想是：先用 first-impact harm bin 做硬门控，只在最小伤害等价类内选择动作；再用 MSRT 预测动作诱导的恢复问题；最后由 RPN 预测五个恢复瓶颈的 signed margins，经 group/scenario-level calibration 后按 CVaR tail risk 选动作。

---

## 1. 工程结构

```text
mrvp_pytorch/
  README.md
  requirements.txt
  configs/default.yaml

  # 论文附录指定的最小可复现入口
  scenario_builder.py          # CARLA-MRVP 数据生成入口
  transition_extractor.py      # contact / boundary-induced transition 提取
  mechanism_labels.py          # z_mech / h_ctx / d_deg 标签提取
  recovery_teachers.py         # degraded MPC / heuristic teacher
  targets.py                   # 五个 bottleneck margin 计算
  train.py                     # MSRT、RPN、pair fine-tuning
  calibrate.py                 # group/scenario conformal lower bounds
  select.py                    # MRVP Algorithm 1 动作选择
  evaluate.py                  # 主表、pair、FRR、coverage 等指标
  train_baseline.py            # direct-action, unstructured latent, scalar baseline
  preprocess_public.py         # 公开数据集 context pretraining 预处理
  pretrain_context.py          # scene/context encoder 预训练
  generate_synthetic.py        # 无 CARLA 的 smoke-test 合成数据

  mrvp/
    data/                      # schema、dataset、same-root pair mining、synthetic generator
    models/                    # MSRT、RPN、baseline networks、context encoder
    carla/                     # CARLA action library、scenario templates、logger、teachers、targets
    public_pretraining/        # nuPlan / Waymo / AV2 / INTERACTION / CommonRoad / highD / nuScenes loaders
    training/                  # checkpoint 和训练循环
    calibration.py             # calibration table 拟合与 lower bound 应用
    selection.py               # harm gate + MSRT samples + RPN + CVaR
    evaluation.py              # 实验指标和 baseline score
```

`docs/paper_method_mapping.md` 给出论文每个公式/附录模块到代码文件的映射。

---

## 2. 环境安装

推荐 Python 3.10。仅跑 PyTorch/synthetic smoke test 不需要安装 CARLA 或公开数据集 SDK。

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

---

## 3. 先跑一个无 CARLA smoke test

这一步生成一个合成 MRVP 数据集，验证 dataset、MSRT、RPN、pair mining、calibration、evaluation、selection 全链路是否可运行。合成数据只用于工程自检，不替代论文实验。

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
  --output runs/synthetic/eval_test.json \
  --split test \
  --torch-threads 1

# 选择某个 root 的动作；root_id 可从 jsonl 里查看
python select.py \
  --data data/synthetic_mrvp.jsonl \
  --root-id syn_000220 \
  --split test \
  --msrt runs/synthetic/msrt.pt \
  --rpn runs/synthetic/rpn_finetuned.pt \
  --calibration runs/synthetic/calibration.json \
  --num-samples 32 \
  --beta 0.9 \
  --torch-threads 1
```

输出文件：

```text
runs/synthetic/msrt.pt
runs/synthetic/rpn.pt
runs/synthetic/rpn_finetuned.pt
runs/synthetic/calibration.json
runs/synthetic/eval_test.json
runs/synthetic/eval_test.csv
```

---

## 4. CARLA-MRVP 数据集构造

### 4.1 启动 CARLA server

```bash
$CARLA_ROOT/CarlaUE4.sh \
  -quality-level=Low \
  -RenderOffScreen \
  -carla-rpc-port=2000
```

本工程的 CARLA generator 会把 world 和 Traffic Manager 设成 synchronous mode，并设置 fixed delta seconds。这样同一个 root scenario 的多个 candidate action 可以在相同 seed / map / traffic / trigger 下 rollout，避免 counterfactual 泄漏。

### 4.2 生成 CARLA-MRVP JSONL

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

每个 root scenario 会扩展成 8 个 candidate actions：

```text
hard_brake, brake_left, brake_right, maintain,
mild_left, mild_right, boundary_side_steer, corridor_side_steer
```

生成流程严格对应论文 Appendix 的 Algorithm “CARLA-MRVP data generation”：

1. `scenario_templates.py` 定义 root scenario：初始 ego state、surrounding actors、map/town、weather、friction、damage/degradation、trigger time、split。
2. `action_library.py` 产生候选紧急动作。
3. `logger.py` 记录 ego/actor state、control、road clearance、secondary clearance、collision sensor event。
4. `transition_extractor.py`：
   - contact：用第一帧 collision event 得到 `t_c`，`x_minus` 是接触前一帧，`x_plus` 是速度/横摆率 reset 稳定后的第一帧；
   - non-contact boundary-critical：用 road/stability/control/secondary/return margins 的 log-sum-exp 选恢复困难入口。
5. `mechanism_labels.py` 生成 `z_mech` 四个分支：geometry、reset、affordance、degradation/uncertainty。
6. `recovery_teachers.py` 从 `(x_plus, d_deg, h_ctx)` 启动 degraded MPC-like teacher。
7. `targets.py` 计算五个 signed margins：secondary collision、road departure、stability、control authority、safe-until-return。
8. 保存一行 JSONL，字段与论文 Table “Core MRVP data schema” 对齐。

### 4.3 数据字段

每一行代表一个 root scenario 下的一个 candidate action：

| 字段 | 含义 |
| --- | --- |
| `root_id` | 同一初始状态、map、traffic seed、weather/friction、trigger 的 counterfactual 组 |
| `action_id`, `action_name`, `action_vec` | 候选紧急动作 |
| `o_hist` | `[T, A, F]` ego 和 surrounding actor history |
| `h_ctx` | route corridor、drivable width、lane boundary、friction、hazard exposure 等 context vector |
| `rho_imp`, `harm_bin` | first-impact harm surrogate 及 monotone bin |
| `x_minus`, `x_plus` | transition 前后状态 |
| `d_deg` | residual steering/brake/throttle authority、delay、friction、damage class |
| `z_mech` | mechanism descriptor，32 维默认布局 |
| `r_star` | 5 个 teacher bottleneck signed margins |
| `b_star`, `s_star` | active bottleneck 和 aggregate margin |
| `calib_group` | contact type、side、friction bin、damage class、density、town 等 calibration group |

root-based split 已经写入 `split=train/val/cal/test`。不要按 action 随机切分，否则同一 root 的 counterfactual 会泄漏到多个 split。

---

## 5. 公开数据集 context / scene encoder 预训练

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

预处理命令：

```bash
# highD: XX_tracks.csv / XX_tracksMeta.csv / XX_recordingMeta.csv
python preprocess_public.py \
  --dataset highd \
  --input /data/highD \
  --output data/pretrain_highd.jsonl \
  --max-records 200000

# INTERACTION: recorded_trackfiles/*/vehicle_tracks_*.csv + maps
python preprocess_public.py \
  --dataset interaction \
  --input /data/INTERACTION \
  --output data/pretrain_interaction.jsonl

# Argoverse 2 Motion Forecasting: scenario parquet
python preprocess_public.py \
  --dataset argoverse2 \
  --input /data/argoverse2/motion-forecasting \
  --output data/pretrain_av2.jsonl

# Waymo Open Motion: Scenario TFRecord / proto，需要 tensorflow + waymo-open-dataset
python preprocess_public.py \
  --dataset waymo \
  --input /data/waymo_open_motion \
  --output data/pretrain_waymo.jsonl

# CommonRoad: XML scenario，需要 commonroad-io
python preprocess_public.py \
  --dataset commonroad \
  --input /data/commonroad \
  --output data/pretrain_commonroad.jsonl

# nuScenes: 需要 nuscenes-devkit
python preprocess_public.py \
  --dataset nuscenes \
  --input /data/sets/nuscenes \
  --version v1.0-trainval \
  --output data/pretrain_nuscenes.jsonl

# nuPlan: 默认用轻量 SQLite fallback；正式实验建议替换为 nuplan-devkit scenario builder
python preprocess_public.py \
  --dataset nuplan \
  --input /data/nuplan \
  --output data/pretrain_nuplan.jsonl
```

合并多个预训练 JSONL 后训练 context encoder：

```bash
cat data/pretrain_highd.jsonl data/pretrain_av2.jsonl data/pretrain_interaction.jsonl > data/pretrain_all.jsonl

python pretrain_context.py \
  --data data/pretrain_all.jsonl \
  --output runs/pretrain/context_encoder.pt \
  --epochs 10 \
  --batch-size 256 \
  --device auto \
  --torch-threads 1
```

当前 `train.py` 会从头训练 MSRT/RPN 的 context encoder。若要使用预训练权重，可在自己的实验脚本中把 `runs/pretrain/context_encoder.pt` 的 state dict load 到 `MSRT.context_encoder` 和 `RecoveryProfileNetwork.context_encoder`，或把 `train.py` 扩展为显式 `--context-init`。

---

## 6. 模型训练

### 6.1 推荐训练顺序

```bash
# 1) 训练 MSRT：transition likelihood + mechanism supervision + physics reset residual
python train.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/carla_mrvp \
  --stage msrt \
  --epochs 30 \
  --batch-size 256 \
  --lr 3e-4 \
  --mixture-count 5 \
  --lambda-mech 1.0 \
  --lambda-phys 0.2 \
  --device auto \
  --torch-threads 1

# 2) 训练 RPN：五个 bottleneck signed margins + active bottleneck CE
python train.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/carla_mrvp \
  --stage rpn \
  --epochs 50 \
  --batch-size 256 \
  --lr 3e-4 \
  --lambda-act 0.5 \
  --lambda-bd 2.0 \
  --sigma-bd 0.5 \
  --device auto \
  --torch-threads 1

# 3) same-root severity-equivalent pair fine-tuning
python train.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/carla_mrvp \
  --stage finetune \
  --rpn-init runs/carla_mrvp/rpn.pt \
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
python train.py --data data/carla_mrvp.jsonl --out-dir runs/carla_mrvp --stage all --epochs 30
```

但正式实验建议分阶段训练，便于检查 transition NLL、mechanism loss、RPN profile loss 和 pair ordering loss。

### 6.2 Calibration

```bash
python calibrate.py \
  --data data/carla_mrvp.jsonl \
  --rpn runs/carla_mrvp/rpn_finetuned.pt \
  --output runs/carla_mrvp/calibration.json \
  --split cal \
  --delta-b 0.02 \
  --n-min 20 \
  --device auto \
  --torch-threads 1
```

实现细节对应 Appendix：对每个 calibration root scenario、group、bottleneck，先在 admissible action set 上取 worst optimistic residual，然后计算 group quantile；小 group 会按 `full -> medium -> coarse -> global` 回退。

---

## 7. 推理 / 动作选择

```bash
python select.py \
  --data data/carla_mrvp.jsonl \
  --root-id carla_0000123 \
  --split test \
  --msrt runs/carla_mrvp/msrt.pt \
  --rpn runs/carla_mrvp/rpn_finetuned.pt \
  --calibration runs/carla_mrvp/calibration.json \
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

## 8. 实验与 baseline

主评估：

```bash
python evaluate.py \
  --data data/carla_mrvp.jsonl \
  --rpn runs/carla_mrvp/rpn_finetuned.pt \
  --calibration runs/carla_mrvp/calibration.json \
  --split test \
  --output runs/carla_mrvp/eval_test.json \
  --device auto \
  --torch-threads 1
```

输出 `eval_test.json` 和 `eval_test.csv`，包含：

- severity envelope violation audit
- severity-matched pair accuracy
- false-recoverable rate, FRR
- worst-bottleneck violation depth
- closed-loop recovery success proxy
- active-bottleneck macro F1
- calibration coverage
- geometry/shift regret proxy

训练论文实验中的网络 baseline：

```bash
# Direct action-to-risk: 不经过 MSRT branches
python train_baseline.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline direct_action_to_risk \
  --epochs 50 \
  --device auto \
  --torch-threads 1

python calibrate.py \
  --data data/carla_mrvp.jsonl \
  --rpn runs/baselines/direct_action_to_risk.pt \
  --model-type direct_action_to_risk \
  --output runs/baselines/direct_calibration.json

python evaluate.py \
  --data data/carla_mrvp.jsonl \
  --rpn runs/baselines/direct_action_to_risk.pt \
  --model-type direct_action_to_risk \
  --calibration runs/baselines/direct_calibration.json \
  --output runs/baselines/direct_eval.json

# Unstructured latent transition: 用 generic latent 替代 geometry/reset/affordance/degradation branches
python train_baseline.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline unstructured_latent \
  --epochs 50

# Scalar recoverability network: 一个 scalar margin broadcast 到五个 bottlenecks
python train_baseline.py \
  --data data/carla_mrvp.jsonl \
  --out-dir runs/baselines \
  --baseline scalar_recoverability \
  --epochs 50
```

`evaluate.py` 同时会报告无需额外训练的 baseline/proxy：

- `Severity_only`
- `Weighted_post_impact_cost`
- `Uncalibrated_MRVP`
- `Scalar_recoverability_network_proxy`
- `Teacher_oracle`

更细的 ablation 可通过下面方式得到：

| Ablation | 命令/改动 |
| --- | --- |
| No harm-equivalence gate | 在 `mrvp/selection.py` 中把 `admissible_indices` 改成返回全部 actions；仅作为不受伦理门控约束的 diagnostic。 |
| No group calibration | `evaluate.py` 中的 `Uncalibrated_MRVP`。 |
| Mean risk instead of CVaR | `select.py --beta 0.0`。 |
| Scalar margin only | `train.py --scalar-rpn` 或 `train_baseline.py --baseline scalar_recoverability`。 |
| No same-root ordering loss | 使用 `runs/.../rpn.pt` 而不是 `rpn_finetuned.pt`。 |
| No mechanism branches | `train_baseline.py --baseline direct_action_to_risk`。 |
| Unstructured latent transition | `train_baseline.py --baseline unstructured_latent`。 |

---

## 9. 单文件 rollout 的调试工具

如果你先保存了 CARLA rollout JSON，可单独跑每个附录组件：

```bash
python transition_extractor.py \
  --rollout data/debug_rollout.json \
  --output data/debug_transition.json

python mechanism_labels.py \
  --rollout data/debug_rollout.json \
  --transition data/debug_transition.json \
  --output data/debug_mechanism.json

python recovery_teachers.py \
  --transition data/debug_transition.json \
  --labels data/debug_mechanism.json \
  --output data/debug_teacher.json \
  --teacher degraded_mpc
```

---

## 10. 重要实现说明

1. **public datasets 只做预训练，不产生完整 MRVP labels。** 论文也指出这些公开数据通常不同时提供 counterfactual emergency actions、post-impact reset、degradation 和 recovery-controller bottleneck margins。
2. **CARLA collision impulse 的精度依赖 simulator 和物理设置。** `transition_extractor.py` 优先用 collision event 的 `normal_impulse`，否则回退到相对速度。
3. **Boundary-critical non-contact 不被伪装成 impact。** 代码使用 recovery-regime entry，即 log-sum-exp margin violation 最大的时刻。
4. **Teacher margins 是 controller-conditioned empirical targets。** 默认 teacher 是无需外部优化器的 degraded MPC-like grid search。正式论文实验可替换为 CasADi/OSQP MPC 或 CBF-QP，但输出仍应遵循 `targets.py` 的五个 signed margins。
5. **Calibration 必须按 root scenario split。** `calibrate.py` 用 root-level worst optimistic residual，避免同一 root 的 action 相关性破坏 coverage。
6. **闭环 CARLA 评估需要你把 `select_action_with_models` 接入 emergency stack。** 本仓库提供离线 `select.py` 和数据生成 closed-loop proxy；真正车控闭环需在 CARLA client 中调用 selector 后执行所选 emergency action，再接 degraded recovery controller。

---

## 11. 参考格式来源

这些链接用于确认 CARLA 和公开数据集的官方/半官方格式，README 中的 preprocessors 按这些格式设计：

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
