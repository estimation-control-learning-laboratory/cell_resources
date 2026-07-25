# Adapting DMAC to a New Plant

This document lists the variables that change when this DMAC
implementation is applied to a plant other than the mass-spring-damper
example, grouped by file, along with the consistency constraints that
must hold between them. Files not listed (`MatrixRLS.m`,
`dmac_identify.m`, `dmac_synthesize_gain.m`, `dmac_control_law.m`,
`run_dmac_simulation.m`, `initialize_logs.m`) are plant-agnostic.

---

## 1. `config_plant.m` — physical system description

| Variable | Meaning | Notes |
|---|---|---|
| `plant.lx` | Full state dimension | Equal to the true state size of the plant. |
| `plant.ly` | Tracked output dimension | Size of the reference-tracking output `y_k`. |
| `plant.lu` | Input dimension | Number of control channels. |
| `plant.lxi` | Measured/partial state dimension | May be smaller than `lx`; this is what enables the partial-state identification path. Equal to `lx` when the full state is measured. |
| physical parameters (`m`, `ks`, `c`, etc.) | Plant-specific constants | Fully plant-dependent; replaced with whatever the new model requires. |
| `plant.A`, `plant.B` | Discrete-time state matrices | Currently produced by `build_discrete_mass_spring_damper.m` via `c2d(..., 'tustin')`. A different plant requires its own continuous-to-discrete build (or a directly specified discrete model), sized `lx x lx` and `lx x lu`. |
| `plant.C` | Output map, `y_k = C * x_k` | Size `ly x lx`. |
| `plant.C_xi` | Measurement map, `xi_k = C_xi * x_k` | Size `lxi x lx`. Encodes which states (or linear combinations) are actually measured. |

## 2. `config_dmac.m` — DMAC/RLS/LQR tuning

| Variable | Meaning | Notes |
|---|---|---|
| `dmac.lambda` | RLS forgetting factor | Typical range `0.9`–`0.999`. Lower values forget faster (more adaptive, higher variance). |
| `dmac.R0` | Initial RLS covariance | Size `(lxi+lu) x (lxi+lu)`. Larger values correspond to faster initial trust in new data. |
| `dmac.Q` | LQR state penalty on the augmented `[xi; q]` state | Size `(lxi+ly) x (lxi+ly)` when `integrator == 'yes'`. Constructed as `blkdiag(Q_xi, Q_q)`, with `Q_xi` (`lxi x lxi`) and `Q_q` (`ly x ly`) tunable independently. `Q_q`'s magnitude interacts with `sim.dt` (see CHANGELOG.md). |
| `dmac.R` | LQR control penalty | Size `lu x lu`. |
| `dmac.v_std` | Excitation std. dev. added to `u_k` | Required for persistence of excitation (PE) so the RLS estimate converges; magnitude is plant/input-scale dependent. |
| `dmac.Cy_xi` | Maps measured state to tracked output, `y_k = Cy_xi * xi_k` | Size `ly x lxi`. Must satisfy `dmac.Cy_xi * plant.C_xi == plant.C` — the tracked output must be recoverable from the measured state alone. Checked in `validate_dmac_config.m`. |
| `dmac.integrator` | `'yes'` or `'no'` | Selects whether the integral-action augmentation is used for reference tracking. |

`dmac.lxi`, `dmac.lu`, `dmac.ly` are copied from `plant` inside
`config_dmac.m` and are not set independently; they follow directly from
`plant`'s dimensions.

## 3. `run_dmac_simulation.m` — simulation-level settings

| Variable | Meaning | Notes |
|---|---|---|
| `sim.N` | Number of simulation steps | Determined by the time needed for RLS convergence and closed-loop settling. |
| `sim.dt` | Sample time | Passed to `config_plant` for discretization; also determines the scaling of the dt-scaled integrator, which affects `dmac.Q`'s integrator block (see CHANGELOG.md). |
| `x(:,1)` | Initial state | Currently `randn(plant.lx,1)`; plant-dependent otherwise. |
| `q(:,1)` | Initial integrator state | Typically `0`. |
| `r` | Reference command | Currently a constant scalar `1`; generalizes to a time-varying signal or vector when `ly > 1` or a trajectory (rather than a step) is required. |
| `randn('state', 2)` | RNG seed | Affects reproducibility of the excitation signal and initial condition only. |

## 4. Cross-file consistency constraints

Checked automatically in `validate_dmac_config.m`:

- `dmac.lxi == plant.lxi`, `dmac.lu == plant.lu`, `dmac.ly == plant.ly`
- `size(plant.C_xi) == [plant.lxi, plant.lx]`
- `size(dmac.Cy_xi) == [plant.ly, plant.lxi]`
- `dmac.Cy_xi * plant.C_xi == plant.C` (within numerical tolerance)
- `size(dmac.R0) == [lxi+lu, lxi+lu]`
- `size(dmac.Q) == [lxi+ly, lxi+ly]` (only when `dmac.integrator == 'yes'`)

Not currently checked:

- Controllability of `(plant.A, plant.B)` and observability of
  `(plant.A, plant.C_xi)`. If either fails to hold for a given plant,
  `dmac_synthesize_gain.m` falls back to the previous gain (see its
  "not controllable" branches) without a dedicated diagnostic message
  distinguishing this from other causes. Checkable directly via
  `rank(ctrb(plant.A,plant.B))` and `rank(obsv(plant.A,plant.C_xi))`.

## 5. Example: 3rd-order plant with partial state measurement

Plant with `lx=3`, measured states `lxi=2` (first two states only),
tracked output `ly=1` (first state):

```matlab
% config_plant.m
plant.lx  = 3;
plant.ly  = 1;
plant.lu  = 1;
plant.lxi = 2;

% A, B: 3x3, 3x1, plant-specific

plant.C    = [1 0 0];              % y_k = x1
plant.C_xi = [1 0 0; 0 1 0];       % xi_k = [x1; x2]
```

```matlab
% config_dmac.m
dmac.Cy_xi = [1 0];                % y_k = xi_k(1), since xi_k = [x1; x2]
```

Consistency check: `dmac.Cy_xi * plant.C_xi = [1 0]*[1 0 0; 0 1 0] = [1 0 0] = plant.C`.
