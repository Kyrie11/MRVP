# 新版论文模块到代码的映射

| 论文模块 / 附录项 | 主要代码 | 说明 |
| --- | --- | --- |
| Harm-equivalence gate | `mrvp/selection.py`, `mrvp/data/schema.py` | `admissible_indices` 只在同一 root 的最小 `harm_bin` 内选动作。 |
| Candidate emergency action library | `mrvp/data/schema.py` | 8 个 action，`action_vec=[steer, throttle, brake, duration]`，默认 duration 为 1s。 |
| Revised row schema | `mrvp/data/schema.py`, `mrvp/data/dataset.py` | Preferred fields：`event_type/event_time/deg/world_plus/teacher_u/teacher_traj/m_star/audit_mech`；兼容 `d_deg/z_mech/r_star`。 |
| MetaDrive/light2d MRVP generation | `scripts/generate_metadrive.py`, `mrvp/data/metadrive_rollout.py` | 对同一 root reset 多个 counterfactual actions，生成 `metadrive_mrvp.jsonl`。 |
| Synthetic smoke-test generation | `scripts/generate_synthetic.py`, `mrvp/data/synthetic.py` | 仅用于工程自检；也输出新版 preferred schema。 |
| MSRT: Action-conditioned Recovery Event Tokenizer | `mrvp/models/msrt.py` | 输入 `(o_hist,h_ctx,x_t,action)`；输出 `event_logits/event_time/x_plus/event_tokens/world_plus/deg`。 |
| MSRT audit/probe fields | `mrvp/models/msrt.py`, `mrvp/data/schema.py` | `z_mech` 作为 `audit_mech` 的兼容/审计表示，不再是 RPN 主输入。 |
| Event token/world losses | `mrvp/models/msrt.py`, `scripts/train.py` | `lambda_event/lambda_token/lambda_world/lambda_deg/lambda_probe/lambda_ctr`。 |
| RPN: Policy-conditioned Recovery Viability Network | `mrvp/models/rpn.py` | 输入 `x_plus, deg, event_tokens, world_plus, h_ctx/o_hist`，解码多条 recovery strategies。 |
| Strategy branches | `mrvp/models/rpn.py` | 输出 `strategy_u`, `strategy_traj`, `branch_margins`；最终取 best branch 的 viability。 |
| RPN losses | `mrvp/models/rpn.py`, `scripts/train.py` | Profile margin loss、active bottleneck CE、strategy imitation、dynamics/control regularization。 |
| Same-root severity-equivalent pair mining | `mrvp/data/pairs.py`, `scripts/train.py --stage finetune` | 只比较同 root 且 harm 接近的 actions。 |
| Group/scenario calibration | `mrvp/calibration.py`, `scripts/calibrate.py` | 按 group/bottleneck 的 conservative lower bound；小组按 hierarchy 回退。 |
| Algorithm 1 / CVaR action selection | `mrvp/selection.py`, `scripts/run_select.py` | `MSRT.sample -> RPN -> calibration -> empirical CVaR`。 |
| Oracle upper-bound evaluation | `scripts/evaluate.py`, `mrvp/evaluation.py` | 使用真实 post-event labels 测 RPN 上界。 |
| Predicted-MRVP evaluation | `scripts/evaluate_predicted_mrvp.py` | 完整预测链路：MSRT 多样本、RPN、calibration、CVaR。 |
| Direct/scalar/unstructured baselines | `mrvp/models/baselines.py`, `scripts/train_baseline.py` | 对比 raw action shortcut 与 scalar recoverability。 |
| Residual-action sufficiency diagnostic | `scripts/diagnose_residual_action.py` | Probe0 看 `x_plus,deg,event_tokens,world_plus,h_ctx`；Probe1 额外看 action。 |
| Public context pretraining | `mrvp/public_pretraining/*`, `scripts/preprocess_public.py`, `scripts/pretrain_context.py` | 只做 scene/context pretraining，不生成 MRVP counterfactual labels。 |
| CARLA future adapter | `mrvp/carla/*`, `scripts/scenario_builder.py` | 保留 CARLA 入口；后续应输出同一新版 JSONL schema。 |
