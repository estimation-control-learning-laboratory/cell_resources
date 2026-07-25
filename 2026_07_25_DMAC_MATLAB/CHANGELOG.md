# DMAC (Dynamic Mode Adaptive Control) - Change Log

**Authors:** Ankit Goel (Assistant Professor)
Mechanical Engineering, University of Maryland, Baltimore County (UMBC)

---

## v11 (2026_07_25) - Computation-time buffer for K
- **Behavior change:** `dmac_identify`/`dmac_synthesize_gain` now use the
  transition `(xi(k-1),u(k-1)) -> xi(k)` instead of
  `(xi(k),u(k)) -> xi(k+1)` when computing the gain `K` applied to
  `xi(k+1)`. Previously, the model update and Riccati solve used data
  that only became available in the same instant `K` was needed for
  `u(k+1)` - i.e. zero computation-time budget. Now that data was
  available a full sample period (`dt`) earlier, giving the RLS
  update + Riccati solve the entire interval to complete before the
  result is used.
- This does **not** change how the feedback law itself is evaluated:
  `u(k+1) = K*xi(k+1) + K_q*q(k+1)` is still a single instantaneous
  matrix multiply, which is the standard (and reasonable) zero-time
  assumption for full-state feedback. The fix only changes which data
  is allowed to inform `K` - not the timing of applying `K` to the
  freshest state.
- No identification update happens at `k=1` (no `k-1` data exists yet);
  `Theta_k`/`P_k`/`K` retain their initial values for that one step,
  matching the guarded pattern that existed as dead/commented-out code
  in the pre-refactor `DMAC_compute_control.m`.
- Minor side benefit: `log.Theta_vec(:,k)` is now labeled more
  correctly - it holds the model estimated from data through step `k`,
  rather than (as before) data through `k+1` logged under index `k`.

## v10 (2026_07_25) - Integrator weight decoupling + dt-scaled integrator
- **Behavior change:** integrator update switched to the dt-scaled form,
  `q(:,k+1) = q(:,k) + e_k*sim.dt`. This makes `q` a proper discrete
  approximation of `∫e dt`, independent of `sim.dt`, but reduces its
  steady-state magnitude by a factor of `dt` relative to the unscaled
  form for the same error history - hence the response got slower when
  this was first tested.
- **Fixed:** `dmac.Q` is now built as `blkdiag(eye(lxi), 100*eye(ly))`
  instead of a single scalar `eye(lxi+lu)`. This both (a) decouples the
  integrator's LQR weight from the xi-state weight, so it can be tuned
  independently, and (b) fixes a latent dimension bug: the integrator
  state `q` has dimension `ly` (built from `y_k = Cy_xi*xi_k`), not
  `lu`. The old sizing only worked because `lu == ly == 1` in this
  plant; a MIMO plant with `lu != ly` would have sized `Q` incorrectly.
- **Rationale for the 100x weight:** since `q_scaled ≈ dt * q_unscaled`
  for the same error trajectory, matching the *quadratic cost* seen by
  the LQR/idare synthesis roughly requires `Q_q_new ≈ Q_q_old / dt^2`.
  With `dt = 0.1` that's a ~100x bump - a smaller linear increase (2x,
  10x) would still leave the closed loop slower than the pre-dt-scaling
  behavior. This is a first-order heuristic, not exact, because the
  closed loop is not perfectly linearly scalable (the control input
  itself changes the error trajectory) - retune from here empirically,
  and check that `dmac.R` still constrains control effort reasonably as
  `Q_q` grows.
- `validate_dmac_config.m` updated to check `dmac.Q` against
  `(lxi+ly)`, not `(lxi+lu)` - this also had the same latent bug and is
  now only enforced when `dmac.integrator == 'yes'` (the augmented
  state only exists in that mode).

## v9 (2026_07_25) - Structural refactor
- Split the monolithic `DMAC_LinSys_MCK_V7.m` / `DMAC_compute_control.m` pair
  into single-responsibility files: `config_plant.m`, `config_dmac.m`,
  `plant_step.m`, `dmac_identify.m`, `dmac_synthesize_gain.m`,
  `dmac_control_law.m`, `validate_dmac_config.m`, plus the top-level
  `run_dmac_simulation.m`.
- **Fixed:** `plant_step` now returns `xi_k = plant.C_xi * x_k` instead of
  `xi_k = x_k`. This was previously masked because `plant.C_xi = eye(2)`
  and `lxi == lx`; it now correctly supports `plant.lxi < plant.lx`
  (partial-state identification/measurement).
- **Fixed:** `dmac.R0` and `dmac.Q` are now sized from `dmac.lxi` (via
  `plant.lxi`), not `plant.lx`. These coincided before only because
  `lxi == lx`.
- **Fixed / clarified:** separated the two previously-conflated "C"
  matrices into `plant.C_xi` (maps full state `x_k` → measured state
  `xi_k`, size `lxi x lx`) and `dmac.Cy_xi` (maps measured state `xi_k` →
  tracked output `y_k`, size `ly x lxi`, used to build the integral
  augmentation). `validate_dmac_config.m` now asserts
  `dmac.Cy_xi * plant.C_xi == plant.C`, so the two stay consistent even
  when `lxi < lx`.
- Preallocated `x, y, xi, q, u` to `sim.N+1` (the loop was writing to
  index `N+1`, silently relying on MATLAB auto-growing the arrays).
- Renamed function-local time-step arguments in the identify/control
  step to match the caller's absolute indices (previously the function's
  internal `k` was actually the script's `k+1`, a readability trap).
- Removed dead/commented-out code paths (old `DMAC_update` branch,
  `k>1` guard, `k < sim.N/2` excitation gating).
- Added missing semicolon on `dmac.integrator = 'yes'` (was printing to
  console every run).

## v8 (2026_04_22) - Fixed integrator
- Added comments, removed unnecessary variables, reorganized.

## v7 (2026_03_26) - Delayed control update
- DMAC now computes `u(k+1)` instead of `u(k)`. Assumes availability of
  `xi(k+1)`, i.e. instantaneous control update at step `k+1`.

## v6 (2026_03_25) - Combined controller update in a single step

## v5 (2026_03_25) - Refactoring
- Encapsulated update in `DMAC_update()`.
- Improved modularity and readability.

## v4 (2026_03_24) - Implementation correction
- Fixed non-causal RLS update.
- Introduced `(phi_{k-1}, xi_k)` causal formulation.

## v3 (2026_03_24) - Initial implementation
- DMAC with RLS-based identification.
- Mass-spring-damper example.
- Integral action for tracking.

## v1.2 (2026_03_25) - Merged implementation (control-synthesis file)

## v1.1 (2026_03_24) - Robustness improvements (control-synthesis file)
- Controllability check added.
- Safe fallback to previous gain.
- Improved dimension handling.

## v1.0 (2026_03_23) - Initial implementation (control-synthesis file)
- Matrix RLS-based identification.
- LQR-based control synthesis.

---

## Open / pending issues
- ~~**Timing (`u_k` depending on `xi_k`):**~~ addressed in v11 - the
  *adaptive gain* `K` is now computed one full sample period ahead of
  when it's applied, giving real computation budget. The instantaneous
  full-state-feedback evaluation `u = K*xi` itself is retained as-is;
  that zero-time assumption is standard for the matrix-multiply step
  and was never the actual issue.
- ~~**Integrator update units**~~ - resolved in v10: switched to
  `e_k*sim.dt` and retuned the integrator's LQR weight accordingly.