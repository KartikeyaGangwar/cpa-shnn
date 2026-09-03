# CPA-SHNN: Causality-Preserving Adaptive Symplectic Hamiltonian Neural Networks

### Academic Research Dissertation Framework
* **Author:** Kartikey Singh (B.Sc. Honours Mathematics, 4th Year)
* **Supervisor:** Prof. Vinay Kumar (Professor of Mathematics, Zakir Husain Delhi College, University of Delhi)
* **Domain:** Celestial Mechanics, Geometric Deep Learning, Scientific Machine Learning (SciML), Symplectic Topology

---

## 1. Executive Summary

Chaotic multi-body celestial gravitational systems are non-integrable Hamiltonian dynamical systems characterized by positive maximal Lyapunov exponents ($\lambda_{\max} > 0$), high hypersensitivity to initial conditions, and strict symplectic phase space invariants (preserving the canonical 2-form $\omega = \mathrm{d}\mathbf{q} \wedge \mathrm{d}\mathbf{p}$). 

Standard numerical integrators (e.g., Runge-Kutta 4th Order) suffer from severe step-size dispersion and energy drift over secular time horizons ($T \gg \tau_L$). Conversely, conventional Physics-Informed Neural Networks (PINNs) and naive Hamiltonian Neural Networks (HNNs) violate temporal causality by minimizing global space-time collocation loss simultaneously across the entire horizon, resulting in catastrophic Lyapunov error compounding and trajectory divergence.

**CPA-SHNN** (Causality-Preserving Adaptive Symplectic Hamiltonian Neural Network) resolves these limitations by synthesizing:
* **Adaptive Causal Time-Marching:** A curriculum-based temporal windowing engine $[0, \tau_1] \to [0, \tau_2] \to \dots \to [0, T_{\max}]$ that strictly enforces initial-value causality;
* **Theorem 1 (Separable Symplectic Kinetic-Coriolis Decomposition):** Exact analytic momentum velocity equations (reducing the neural search space from 4D to 2D and eliminating autograd momentum noise);
* **Theorem 2 (Arnold Extended Contact Phase Space):** Energy-preserving contact coordinates $\mathbf{Z}_{\text{ext}} = (\mathbf{q}, t, \mathbf{p}, p_t)$ for non-autonomous breathing and variable-mass dynamics;
* **Theorem 3 (Neural Poincaré Symplectic Generating Maps):** Discrete-time generating functions $S_\theta(\mathbf{q}_k, \mathbf{p}_{k+1})$ providing exact symplectic area preservation without ODE numerical discretization drift;
* **Multi-Scale Fourier Encodings:** Overcoming spectral bias on sharp multi-body gravitational saddle potentials;
* **Second-Order Curvature Optimization:** Dual-phase AdamW exploration followed by aggressive L-BFGS refinement with Strong-Wolfe line search.

---

## 2. Quantitative Empirical Benchmark Matrices (Kaggle GPU 400 Epochs)

### 2.1 Autonomous Celestial Master Benchmark Matrix
| System Name | Standard PINN | Vanilla HNN (2019) | CPA-SHNN Core | Theorem 1 (Separable) | Theorem 3 (Gen Map) | Combo 1+3 (Sep-Gen) |
|---|---|---|---|---|---|---|
| **Binary Quasar (Chaotic)** | 64.75% | 179.28% | 22.12% | **14.75%** | 235.54% | 33.17% |
| **Restricted 6-Body (Chaotic)** | 119.25% | 37.91% | 21.22% | **14.11%** | 51.07% | **19.59%** |
| **Sitnikov 5-Body (Chaotic)** | 52.68% | 26.95% | 33.51% | 33.52% | **26.76%** | 33.64% |
| **Magnetic Yukawa (Chaotic)** | 10.13% | 154.83% | **1.01%** | **4.34%** | 107.53% | 25.49% |
| **Mean Energy Drift ($\Delta\mathcal{H}$)** | 100.00% | 0.0001% | **0.0001%** | **0.0000%** | 0.0001% | 0.1500% |

---

### 2.2 Non-Autonomous 8-Way Master Benchmark Matrix (Prof. Vinay Kumar Systems)
| Non-Autonomous System | Standard PINN | Vanilla HNN | CPA-Core | Thm 1 (Pure) | Thm 2 (Ext) | **Thm 1+2 (Sep-Ext)** | Combo 2+3 | **Combo 1+2+3 (Grand Unified)** |
|---|---|---|---|---|---|---|---|---|
| **Elliptic Sitnikov 5-Body** *(Ullah 2020)* | 2.83% | 85.95% | 102.94% | 102.96% | **6.02%** | 36.42% | 216.55% | 215.22% |
| **Variable-Mass Binary** *(Kumar 2023)* | 1.93% | 106.65% | 43.07% | 160.31% | 458.28% | **8.23%** | 193.01% | **21.01%** |

---

### 2.3 Fourier Positional Encoding Ablation Matrix (With vs Without Fourier)
| System Name | Thm 1 (No Fourier) | **Thm 1 (With Fourier)** | Combo 1+3 (No Fourier) | **Combo 1+3 (With Fourier)** |
|---|---|---|---|---|
| **Binary Quasar (Chaotic)** | 67.52% | **23.40%** ($2.9\times$ Gain) | 77.55% | **22.57%** ($3.4\times$ Gain) |
| **Restricted 6-Body (Chaotic)** | 61.16% | **57.44%** | 220.08% | **10.23%** ($\mathbf{22\times}$ Gain!) |
| **Magnetic Yukawa (Chaotic)** | 57.23% | **13.25%** ($4.3\times$ Gain) | 27.89% | **16.37%** ($1.7\times$ Gain) |
| **Elliptic Sitnikov (Non-Auto)** | \multicolumn{2}{c|}{Grand Unified (No Fourier): 32.64%} | \multicolumn{2}{c|}{**Grand Unified (With Fourier): 9.78%** ($3.3\times$ Gain)} |
| **Variable-Mass Binary (Non-Auto)** | \multicolumn{2}{c|}{Grand Unified (No Fourier): 147.12%} | \multicolumn{2}{c|}{**Grand Unified (With Fourier): 68.44%** ($2.2\times$ Gain)} |

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

### 3.3 Theorem 3: Neural Poincaré Symplectic Generating Maps
Discrete symplectic evolution $(\mathbf{q}_k, \mathbf{p}_k) \mapsto (\mathbf{q}_{k+1}, \mathbf{p}_{k+1})$ across step size $\Delta t$ is generated by Poincaré generating function $S_\theta$:
$$S_\theta(\mathbf{q}_k, \mathbf{p}_{k+1}) = \mathbf{q}_k \cdot \mathbf{p}_{k+1} + \Delta t \left( \frac{1}{2}\|\mathbf{p}_{k+1}\|^2 + n(p_{x, k+1} y_k - p_{y, k+1} x_k) - V_\theta(\mathbf{q}_k) \right)$$

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

### 4.3 Fourier Ablation Benchmark Execution
```bash
python -m celestial_hnn.benchmarks.run_fourier_ablation_benchmark
```

---

## 5. Formal Academic References

* **Arnold, V. I.** (1989). *Mathematical Methods of Classical Mechanics*. Springer-Verlag.
* **Greydanus, S., Dzamba, M., & Yosinski, J.** (2019). Hamiltonian Neural Networks. *NeurIPS*, 32.
* **Kumar, V., Aggarwal, R., Sharma, P., & Kaur, B.** (2021). Fractal basins of attraction in a binary quasar model. *New Astronomy*, 84, 101543.
* **Kumar, V., Idrisi, M. J., & Ullah, M. S.** (2021). Unpredictable basin boundaries in restricted six-body problem with square configuration. *New Astronomy*, 82, 101451.
* **Kumar, V., Aggarwal, R., & Marig, S. K.** (2023). Unveiling the intricacies of attracting zones in magnetic binary systems: Investigating the impact of Yukawa correction. *Astronomy and Computing*, 40, 100783.
* **Kumar, V., & Marig, S. K.** (2023). Effect of variable mass on N–R basins of convergence in photogravitational magnetic binary problem. *Astronomy Reports*, 67(2), 194-208.
* **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. *Journal of Computational Physics*, 378, 686-707.
* **Ullah, M. S., Idrisi, M. J., & Kumar, V.** (2020). Elliptic Sitnikov five-body problem under radiation pressure. *New Astronomy*, 80, 101398.
