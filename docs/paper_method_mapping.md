# MRVP paper-to-code mapping

| Paper item | Code file | Notes |
| --- | --- | --- |
| Harm-equivalence gate, Eq. (1) | `mrvp/selection.py`, `mrvp/data/schema.py` | `admissible_indices` keeps only the root-level minimum harm bin. |
| CARLA-MRVP root scenarios | `mrvp/carla/scenario_templates.py`, `mrvp/carla/scenario_builder.py` | Root split is scenario-based; all action counterfactuals stay in one split. |
| Candidate action library | `mrvp/carla/action_library.py` | Eight actions from the appendix. |
| Contact transition extraction | `mrvp/carla/transition_extractor.py` | Uses first collision event and velocity/yaw-rate stabilization. |
| Boundary-induced transition | `mrvp/carla/transition_extractor.py` | Uses log-sum-exp over road/stability/control/secondary/return margins. |
| Mechanism labels, Table in appendix | `mrvp/carla/mechanism_labels.py` | Geometry, reset, affordance, degradation and uncertainty branches. |
| Recovery teachers and targets | `mrvp/carla/recovery_teachers.py`, `mrvp/carla/targets.py` | Degraded MPC-like grid teacher and heuristic teacher. |
| MSRT architecture/loss | `mrvp/models/msrt.py` | Context/action encoders, mechanism heads, mixture decoder, physics-reset prior. |
| RPN architecture/loss | `mrvp/models/rpn.py` | Bottleneck-specific signed margins, active-bottleneck CE, boundary weighting. |
| Same-root pair mining | `mrvp/data/pairs.py`, `train.py --stage finetune` | Pairs share root_id and harm_bin. |
| Calibration | `mrvp/calibration.py`, `calibrate.py` | Root-level worst optimistic residuals with group fallback. |
| CVaR selection, Algorithm 1 | `mrvp/selection.py`, `select.py` | Samples MSRT transitions and selects lowest calibrated tail risk. |
| Experiments and metrics | `mrvp/evaluation.py`, `evaluate.py` | Main metrics and baseline proxies. |
| Public pretraining | `mrvp/public_pretraining/*`, `preprocess_public.py`, `pretrain_context.py` | Converts nuPlan, Waymo, Argoverse2, INTERACTION, CommonRoad, highD, nuScenes to context-pretrain JSONL. |
