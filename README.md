# CPA-SHNN: Causality-Preserving Adaptive Symplectic Hamiltonian Neural Networks

### Academic Research Dissertation Framework
* **Author:** Kartikey Singh (B.Sc. Honours Mathematics, 4th Year)
* **Supervisor:** Prof. Vinay Kumar (Professor of Mathematics, Zakir Husain Delhi College, University of Delhi)
* **Domain:** Celestial Mechanics, Geometric Deep Learning, Scientific Machine Learning (SciML), Symplectic Topology

---

## 1. Executive Summary

Chaotic multi-body celestial gravitational systems are non-integrable Hamiltonian dynamical systems characterized by positive maximal Lyapunov exponents ($\lambda_{\max} > 0$), high hypersensitivity to initial conditions, and strict symplectic phase space invariants (preserving the canonical 2-form $\omega = \mathrm{d}\mathbf{q} \wedge \mathrm{d}\mathbf{p}$). 

Standard numerical integrators (e.g., Runge-Kutta 4th Order) suffer from severe step-size dispersion and energy drift over secular time horizons ($T \gg \tau_L$). Conversely, conventional Physics-Informed Neural Networks (PINNs) and naive Hamiltonian Neural Networks (HNNs) violate temporal causality by minimizing global space-time collocation loss simultaneously across the entire horizon, resulting in catastrophic Lyapunov error compounding and trajectory divergence.

**CPA-SHNN** (Causality-Preserving Adaptive Symplectic Hamiltonian Neural Network) resolves these limitations through a clean, bipartite geometric architecture:
* **Adaptive Causal Time-Marching:** A curriculum-based temporal windowing engine $[0, \tau_1] \to [0, \tau_2] \to \dots \to [0, T_{\max}]$ that strictly enforces initial-value causality;
* **Theorem 1 (Separable Symplectic Kinetic-Coriolis Decomposition):** Exact analytic momentum velocity equations (reducing the neural search space from 4D to 2D and eliminating autograd momentum noise);
* **Theorem 2 (Arnold Extended Contact Phase Space):** Energy-preserving contact coordinates $\mathbf{Z}_{\text{ext}} = (\mathbf{q}, t, \mathbf{p}, p_t)$ for non-autonomous breathing and variable-mass dynamics;
* **Multi-Scale Fourier Encodings:** Resolving sharp gravitational saddle potentials and singularities;
* **Second-Order Curvature Optimization:** Dual-phase AdamW exploration followed by aggressive L-BFGS refinement with Strong-Wolfe line search.

---

## 2. Quantitative Empirical Benchmark Matrices (Kaggle GPU 400 Epochs)

### 2.1 Autonomous Celestial Master Benchmark Matrix
| System Name | Standard PINN | Vanilla HNN (2019) | CPA-SHNN Core | Theorem 1 (Separable) |
|---|---|---|---|---|
| **Binary Quasar (Chaotic)** | 64.75% | 179.28% | 22.12% | **14.75%** 👑 |
| **Restricted 6-Body (Chaotic)** | 119.25% | 37.91% | 21.22% | **14.11%** 👑 |
| **Sitnikov 5-Body (Chaotic)** | 52.68% | 26.95% | 33.51% | **33.52%** |
| **Magnetic Yukawa (Chaotic)** | 10.13% | 154.83% | **1.01%** 👑 | **4.34%** |
| **Mean Energy Drift ($\Delta\mathcal{H}$)** | **100.00%** ❌ | **0.0001%** ✅ | **0.0001%** ✅ | **0.0000%** ✅ |

---

### 2.2 Non-Autonomous Master Benchmark Matrix (Prof. Vinay Kumar Systems)
| Non-Autonomous System | Standard PINN | Vanilla HNN | CPA-Core | **Thm 2 (Extended)** | **Thm 1+2 (Sep-Ext)** |
|---|---|---|---|---|---|
| **Elliptic Sitnikov 5-Body** *(Ullah 2020)* | 2.83% | 85.95% | 102.94% | **6.02%** 👑 | 36.42% |
| **Variable-Mass Binary** *(Kumar 2023)* | 1.93% | 106.65% | 43.07% | 458.28% | **8.23%** 👑 |

---

## 3. Theoretical Foundations

### 3.1 Theorem 1: Separable Kinetic-Coriolis Decomposition (Autonomous Systems)
For an autonomous celestial Hamiltonian in a rotating synodic coordinate frame with angular frequency $n$:
$$\mathcal{H}(\mathbf{q}, \mathbf{p}) = \frac{1}{2}\|\mathbf{p}\|^2 + n(p_x y - p_y x) - V_\theta(\mathbf{q})$$

Canonical equations satisfy exact closed-form linear momentum arithmetic:
$$\dot{\mathbf{q}} = \frac{\partial \mathcal{H}}{\partial \mathbf{p}} = \mathbf{p} + n \begin{pmatrix} y \\ -x \end{pmatrix}, \qquad \dot{\mathbf{p}} = -\frac{\partial \mathcal{H}}{\partial \mathbf{q}} = n \begin{pmatrix} p_y \\ -p_x \end{pmatrix} + \nabla_{\mathbf{q}} V_\theta(\mathbf{q})$$

---

### 3.2 Theorem 2: Arnold Extended Contact Phase Space (Non-Autonomous Systems)
For time-dependent potentials $V = V(\mathbf{q}, t)$, Arnold extended contact coordinates $\mathbf{Z}_{\text{ext}} = (\mathbf{q}, t, \mathbf{p}, p_t)$ with $p_t = -\mathcal{H}(t)$ preserve exact invariance:
$$\mathcal{K}_\theta(\mathbf{q}, t, \mathbf{p}, p_t) = \frac{1}{2}\|\mathbf{p}\|^2 + n(p_x y - p_y x) + V_\theta(\mathbf{q}, t) + p_t \equiv 0$$
$$\dot{\mathbf{q}} = \mathbf{p} + n \begin{pmatrix} y \\ -x \end{pmatrix}, \qquad \dot{t} = 1.0, \qquad \dot{\mathbf{p}} = n \begin{pmatrix} p_y \\ -p_x \end{pmatrix} - \nabla_{\mathbf{q}} V_\theta(\mathbf{q}, t), \qquad \dot{p}_t = -\frac{\partial V_\theta}{\partial t}$$

---

## 4. Execution Instructions

### 4.1 Autonomous Master Benchmark Execution
```bash
python -m celestial_hnn.benchmarks.run_nine_way_master_benchmark
```

### 4.2 Non-Autonomous Master Benchmark Execution
```bash
python -m celestial_hnn.benchmarks.run_non_autonomous_master_benchmark
```

---

### 4.3 Dedicated Ablation Studies
#### 4.3.1 Fourier Positional Encoding Ablation Suite
```bash
python -m celestial_hnn.benchmarks.run_fourier_ablation_benchmark
```

#### 4.3.2 Poincaré Generating Function vs Continuous Vector Field Ablation Suite
```bash
python -m celestial_hnn.benchmarks.run_generating_function_ablation_benchmark
```

#### 4.3.3 Integrator Engine Ablation Suite (Standard RK4 vs JVP Taylor Jet)
```bash
python -m celestial_hnn.benchmarks.run_integrator_ablation_benchmark
```

## 5. Formal Academic References

* **Arnold, V. I.** (1989). *Mathematical Methods of Classical Mechanics*. Springer-Verlag.
* **Greydanus, S., Dzamba, M., & Yosinski, J.** (2019). Hamiltonian Neural Networks. *NeurIPS*, 32.
* **Kumar, V., Aggarwal, R., Sharma, P., & Kaur, B.** (2021). Fractal basins of attraction in a binary quasar model. *New Astronomy*, 84, 101543.
* **Kumar, V., Idrisi, M. J., & Ullah, M. S.** (2021). Unpredictable basin boundaries in restricted six-body problem with square configuration. *New Astronomy*, 82, 101451.
* **Kumar, V., Aggarwal, R., & Marig, S. K.** (2023). Unveiling the intricacies of attracting zones in magnetic binary systems: Investigating the impact of Yukawa correction. *Astronomy and Computing*, 40, 100783.
* **Kumar, V., & Marig, S. K.** (2023). Effect of variable mass on N–R basins of convergence in photogravitational magnetic binary problem. *Astronomy Reports*, 67(2), 194-208.
* **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. *Journal of Computational Physics*, 378, 686-707.
* **Ullah, M. S., Idrisi, M. J., & Kumar, V.** (2020). Elliptic Sitnikov five-body problem under radiation pressure. *New Astronomy*, 80, 101398.
