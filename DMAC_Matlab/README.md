# DMAC Matlab
This repository contains example implementations of DMAC (Dynamic Mode Adaptive Control) using online system identification and adaptive LQR control. The method combines online matrix-based Recursive Least Squares (RLS) with optimal control to achieve reference tracking without prior knowledge of the system model.

Two benchmark examples are provided:
- A linear Mass–Spring–Damper (MCK) system
- A nonlinear Van der Pol (VDP) oscillator

extract_A_B_DMD:
Splits the identified parameter matrix into the system matrix A and input matrix B for control design.

generate_augmented_A_B_DMD:
Creates an augmented system that adds integral action to enable reference tracking and eliminate steady-state error.

RLS_update_DMD:
Updates the system model online using measured data and provides a prediction error to monitor model accuracy.
