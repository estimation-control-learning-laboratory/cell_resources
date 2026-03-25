# Dynamic Mode Adaptive Control (DMAC) – MATLAB

This repository contains MATLAB implementations of **Dynamic Mode Adaptive Control (DMAC)**, a data-driven adaptive control framework that combines online system identification with optimal control.

The method uses measured state and input data to:
- Identify a local linear model online
- Compute feedback gains via LQR
- Achieve reference tracking without a prior model

---

## Examples

Two benchmark systems are included:

- **Mass–Spring–Damper (MCK) system**  
  Linear system used to demonstrate convergence and tracking performance

- **Van der Pol (VDP) oscillator**  
  Nonlinear system illustrating adaptation under changing dynamics

---

## Key Functions

- **`DMAC_update`**  
  Performs one step of parameter estimation and control update

- **`MatrixRLS`**  
  Online system identification using Recursive Least Squares (RLS)

- **`compute_DMAC_control`**  
  Computes feedback gains from the estimated model

- **`extract_A_B_from_Theta`**  
  Extracts system matrices from the parameter estimate

- **`generate_augmented_A_B_DMAC`**  
  Constructs augmented system for reference tracking

- **`plot_DMAC_results`**  
  Generates standard plots for tracking, control effort, and parameter evolution

---

## Features

- Fully **data-driven** (no prior model required)
- Online **adaptive control**
- Supports **linear and nonlinear systems**
- Modular and easy to extend

---

## Requirements

- MATLAB  
- Control System Toolbox (`idare`, `ctrb`, `c2d`)

---

## Authors
**Parham Oveissi**, PhD Candidate
**Dr. Juan Paredes**, Research Fellow  
**Dr. Ankit Goel**, Assistant Professor  
Mechanical Engineering  
University of Maryland, Baltimore County (UMBC)

---

## License

[Add license here]