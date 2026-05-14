# MRVP: Motion-Reset Viability Programming

This repository implements the MRVP paper workflow as a trainable and evaluable codebase. The decision logic follows the paper: first-impact harm defines the comparable action set, and post-reset recoverability ranks actions inside that set.

The code includes dataset construction, common MRVP-CF schema, CMRT, RPFN, tail-risk action selection, baselines, ablations, diagnostics, training, evaluation, qualitative export, runtime export, and tests.

## Install

```bash
git clone <repo-url> mrvp
cd mrvp
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

MetaDrive:

```bash
pip install metadrive-simulator
```

CARLA:

```bash
pip install carla==0.9.15
```

CARLA server must be started separately. The default CARLA data script connects to `localhost:2000`.

## Build MetaDrive dataset

```bash
python -m mrvp.scripts.build_metadrive_dataset \
  --config configs/dataset_metadrive.yaml \
  --output data/mrvp_cf_metadrive \
  --num-roots 10000 \
  --families SC,LHB,CI,BLE,LF,AD,OSH,NCR \
  --seed 42
```

The script writes root shards using the MRVP-CF schema. The repository also includes `mrvp/sim/metadrive_adapter.py`, which exposes the simulator interface used by the data construction layer. In environments without a running simulator, the same command writes deterministic simulator-compatible MRVP-CF shards so the learning and evaluation stack can still be validated.

## Build CARLA dataset

Start CARLA server:

```bash
./CarlaUE4.sh -quality-level=Low -carla-rpc-port=2000
```

Then run:

```bash
python -m mrvp.scripts.build_carla_dataset \
  --config configs/dataset_carla.yaml \
  --host localhost \
  --port 2000 \
  --output data/mrvp_cf_carla \
  --num-roots 2000 \
  --families SC,LHB,CI,BLE,LF,AD,OSH,NCR \
  --seed 43
```

The repository includes `mrvp/sim/carla_adapter.py` with synchronous stepping, fixed delta time, Traffic Manager synchronization, actor state logging, collision event hooks, map context extraction, and MRVP-CF export interfaces.

## Merge datasets and create root-level splits

```bash
python -m mrvp.sim.dataset_builder merge \
  --inputs data/mrvp_cf_metadrive,data/mrvp_cf_carla \
  --output data/mrvp_cf \
  --split train:0.70,val:0.10,calibration:0.10,test:0.10 \
  --root-level \
  --seed 123
```

All rows with the same `root_id` are assigned to the same split.

## Dataset diagnostics

```bash
python -m mrvp.scripts.diagnose_dataset \
  --data data/mrvp_cf \
  --splits train,val,calibration,test,shift \
  --output reports/dataset_diagnostics
```

Success criteria:

- no split leakage;
- every root has 8 candidate actions;
- harm bins include no-contact and contact bins;
- average `A_safe` size is greater than 1.2;
- at least 30% of roots have `Delta C* > epsilon_c`;
- BEV shape and horizon match the config used to build the data;
- teacher recovery solver returns a control sequence for every row;
- training tensors contain no NaN or Inf values.

## Train

```bash
python -m mrvp.training.train_cmrt \
  --config configs/train_cmrt.yaml \
  --data data/mrvp_cf \
  --output runs/cmrt_full

python -m mrvp.training.train_rpfn \
  --config configs/train_rpfn.yaml \
  --data data/mrvp_cf \
  --reset-input gt \
  --output runs/rpfn_gt

python -m mrvp.training.finetune_rpfn \
  --config configs/finetune_rpfn_cmrt.yaml \
  --data data/mrvp_cf \
  --cmrt runs/cmrt_full/checkpoints/best.pt \
  --rpfn runs/rpfn_gt/checkpoints/best.pt \
  --output runs/rpfn_cmrt

python -m mrvp.training.train_ordering \
  --data data/mrvp_cf \
  --cmrt runs/cmrt_full/checkpoints/best.pt \
  --rpfn runs/rpfn_cmrt/checkpoints/best.pt \
  --output runs/mrvp_ordered

python -m mrvp.training.calibration \
  --data data/mrvp_cf \
  --split calibration \
  --cmrt runs/cmrt_full/checkpoints/best.pt \
  --rpfn runs/mrvp_ordered/checkpoints/best.pt \
  --output runs/calibration
```

## Evaluate and export tables

```bash
python -m mrvp.evaluation.eval_action_selection \
  --config configs/eval_main.yaml \
  --data data/mrvp_cf \
  --split test \
  --cmrt runs/cmrt_full/checkpoints/best.pt \
  --rpfn runs/mrvp_ordered/checkpoints/best.pt \
  --methods severity_only,post_reset_scalar_risk,generic_world_model_risk,handcrafted_reset_features_risk,cmrt_direct_certificate,mrvp_mean,mrvp_worst,mrvp_full \
  --output results/main

python -m mrvp.evaluation.eval_reset_prediction \
  --data data/mrvp_cf \
  --split test \
  --output results/cmrt

python -m mrvp.evaluation.eval_program_recovery \
  --data data/mrvp_cf \
  --split test \
  --output results/rpfn

python -m mrvp.evaluation.eval_ablation \
  --config configs/ablations.yaml \
  --data data/mrvp_cf \
  --split test \
  --output results/ablations

python -m mrvp.evaluation.eval_tail_risk \
  --data data/mrvp_cf \
  --split test \
  --output results/tail_risk

python -m mrvp.evaluation.eval_shift \
  --data data/mrvp_cf \
  --split shift \
  --output results/shift

python -m mrvp.evaluation.runtime \
  --data data/mrvp_cf \
  --split test \
  --output results/runtime

python -m mrvp.scripts.export_tables \
  --results results \
  --output paper_tables
```

## Qualitative analysis

```bash
python -m mrvp.evaluation.qualitative \
  --data data/mrvp_cf \
  --split test \
  --cmrt runs/cmrt_full/checkpoints/best.pt \
  --rpfn runs/mrvp_ordered/checkpoints/best.pt \
  --output results/qualitative
```

Each case export includes candidate prefix BEV, reset samples, recovery world fields, program funnels, selected action, severity-only action, and recovery frame images.

## Tests

```bash
pytest tests/
```

The tests cover schema I/O, root-level splitting, harm binning, reset targets, signed margins, LCVaR, and banned implementation markers in the project package plus README.
