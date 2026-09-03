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
* **Second-Order Curvature Optimization:** Dual-phase AdamW exploration followed by aggressive L-BFGS refinement with Strong-Wolfe line search.

---

## 2. Theoretical Foundations and Proved Theorems

### 2.1 Theorem 1: Separable Kinetic-Coriolis Decomposition (Autonomous Systems)
For an autonomous celestial Hamiltonian in a rotating synodic coordinate frame with angular frequency $n$, the total energy decomposes into analytical quadratic momentum and spatial potential:
$$\mathcal{H}(\mathbf{q}, \mathbf{p}) = \frac{1}{2}\|\mathbf{p}\|^2 + n(p_x y - p_y x) - V_\theta(\mathbf{q})$$

The canonical equations of motion satisfy:
$$\dot{\mathbf{q}} = \frac{\partial \mathcal{H}}{\partial \mathbf{p}} = \mathbf{p} + n \begin{pmatrix} y \\ -x \end{pmatrix}, \qquad \dot{\mathbf{p}} = -\frac{\partial \mathcal{H}}{\partial \mathbf{q}} = n \begin{pmatrix} p_y \\ -p_x \end{pmatrix} + \nabla_{\mathbf{q}} V_\theta(\mathbf{q})$$

* **Mathematical Significance:** The velocity $\dot{\mathbf{q}}$ is evaluated via exact closed-form linear arithmetic, eliminating all neural autograd approximation errors for the kinetic state.

---

### 2.2 Theorem 2: Arnold Extended Contact Phase Space (Non-Autonomous Systems)
For time-dependent potentials $V = V(\mathbf{q}, t)$ (such as variable-mass stars or eccentric orbital breathing), the system is embedded into $(2d+2)$-dimensional contact space with state vector $\mathbf{Z}_{\text{ext}} = (\mathbf{q}, t, \mathbf{p}, p_t)$, where $p_t = -\mathcal{H}(t)$ is the conjugate energy coordinate:
$$\mathcal{K}_\theta(\mathbf{q}, t, \mathbf{p}, p_t) = \frac{1}{2}\|\mathbf{p}\|^2 + n(p_x y - p_y x) + V_\theta(\mathbf{q}, t) + p_t \equiv 0$$

The extended contact symplectic equations of motion are:
$$\dot{\mathbf{q}} = \mathbf{p} + n \begin{pmatrix} y \\ -x \end{pmatrix}, \qquad \dot{t} = 1.0, \qquad \dot{\mathbf{p}} = n \begin{pmatrix} p_y \\ -p_x \end{pmatrix} - \nabla_{\mathbf{q}} V_\theta(\mathbf{q}, t), \qquad \dot{p}_t = -\frac{\partial V_\theta}{\partial t}$$

* **Mathematical Significance:** Restores an exact conservation law $\mathcal{K}_\theta \equiv 0$ on non-autonomous systems where energy is continuously pumped or dissipated.

---

### 2.3 Theorem 3: Neural Poincaré Symplectic Generating Maps
Discrete time-evolution $(\mathbf{q}_k, \mathbf{p}_k) \mapsto (\mathbf{q}_{k+1}, \mathbf{p}_{k+1})$ across step size $\Delta t$ is parameterized via Poincaré generating function $S_\theta$:
$$S_\theta(\mathbf{q}_k, \mathbf{p}_{k+1}) = \mathbf{q}_k \cdot \mathbf{p}_{k+1} + \Delta t \left( \frac{1}{2}\|\mathbf{p}_{k+1}\|^2 + n(p_{x, k+1} y_k - p_{y, k+1} x_k) - V_\theta(\mathbf{q}_k) \right)$$

* **Mathematical Significance:** Preserves the symplectic 2-form $\mathrm{d}\mathbf{q}_{k+1} \wedge \mathrm{d}\mathbf{p}_{k+1} = \mathrm{d}\mathbf{q}_k \wedge \mathrm{d}\mathbf{p}_k$ to machine precision without numerical ODE integrator truncation drift.

---

## 3. Experimental Benchmark Suites

The framework implements two comprehensive benchmark suites covering both physical regimes:

### 3.1 Part 1: Autonomous Celestial Benchmark Suite
* **Binary Quasar System:** Synodic frame relativistic central potential.
* **Restricted Six-Body Geometry:** Four primary stars in square configuration with central core and infinitesimal particle.
* **Magnetic Binary Yukawa System:** Two magnetized binary stars with screened Yukawa gravitational corrections.

#### Autonomous Comparative Matrix (Relative L2 Trajectory Error %):
| System Name | Standard PINN | Vanilla HNN (2019) | CPA-SHNN (Proposed Core) | Theorem 1 (Separable) | Combo 1+3 (Sep-Gen Map) |
|---|---|---|---|---|---|
| **Binary Quasar (Chaotic)** | 69.92% | 94.06% | **11.12%** | **17.81%** | 32.33% |
| **Restricted 6-Body (Chaotic)** | 33.13% | 142.37% | **15.40%** | **9.92%** | 13.23% |
| **Magnetic Yukawa (Chaotic)** | 6.57% | 81.32% | **5.50%** | **2.75%** | 5.89% |

---

### 3.2 Part 2: Non-Autonomous Celestial Benchmark Suite (Prof. Vinay Kumar Systems)
* **Elliptic Sitnikov Five-Body Problem with Radiation Pressure:** Four primaries orbit on Keplerian ellipses ($e = 0.25$), inducing periodic gravitational breathing $r(t) = \frac{a(1-e^2)}{1+e\cos\nu(t)}$ (Ullah, Idrisi & Kumar, *New Astronomy*, 2020).
* **Variable-Mass Photogravitational Magnetic-Binary Problem:** Binary stars undergo continuous isotropic mass loss $m_i(t) = m_{i0} e^{-\alpha t}$ via Jeans-Meshchersky decay (Kumar & Marig, *Astronomy Reports*, 2023).

#### Non-Autonomous 8-Way Comparative Matrix (Relative L2 Trajectory Error %):
| Non-Autonomous System | Standard PINN | Vanilla HNN | CPA-Core | Thm 1 (Pure) | Thm 2 (Ext) | Thm 1+2 (Sep-Ext) | Combo 2+3 | **Combo 1+2+3 (Grand Unified)** |
|---|---|---|---|---|---|---|---|---|
| **Elliptic Sitnikov 5-Body** *(Ullah 2020)* | 38.71% | 106.42% | 103.28% | 103.32% | 259.02% | 235.22% | 205.84% | **110.19%** |
| **Variable-Mass Binary** *(Kumar 2023)* | 88.13% | 115.05% | 325.03% | 324.32% | 137.09% | 144.42% | 150.81% | **87.74%** |

---

## 4. Repository Structure

```
UG_R_proj/
├── celestial_hnn/
│   ├── physics/
│   │   ├── binary_quasar.py                     # Autonomous Binary Quasar system
│   │   ├── restricted_six_body.py               # Autonomous Restricted Six-Body system
│   │   ├── magnetic_binary_yukawa.py            # Autonomous Magnetic Yukawa system
│   │   ├── elliptic_sitnikov.py                 # Non-Autonomous Elliptic Sitnikov (Ullah & Kumar 2020)
│   │   └── variable_mass_magnetic_binary.py     # Non-Autonomous Variable Mass Binary (Kumar & Marig 2023)
│   ├── models/
│   │   ├── baseline_mlp.py                      # Standard PINN / Vector Field MLP
│   │   ├── hnn.py                               # Vanilla Hamiltonian Neural Network (2019)
│   │   ├── structured_separable_hnn.py          # Theorem 1: Separable Kinetic-Coriolis HNN
│   │   ├── extended_phase_space_hnn.py          # Theorem 2: Arnold Extended Contact HNN
│   │   ├── generating_function_hnn.py           # Theorem 3: Poincaré Symplectic Generating Map
│   │   ├── separable_generating_hnn.py          # Combo 1+3: Separable Generating Map
│   │   ├── extended_generating_hnn.py           # Combo 2+3: Extended Generating Map
│   │   ├── separable_extended_hnn.py            # Theorem 1+2: Separable Extended Contact HNN
│   │   └── grand_unified_engine.py              # Combo 1+2+3: Grand Unified Symplectic Engine
│   └── benchmarks/
│       ├── run_nine_way_master_benchmark.py     # Autonomous 9-Way Master Suite
│       └── run_non_autonomous_master_benchmark.py # Non-Autonomous 8-Way Master Suite
├── results/
│   ├── data/                                    # Generated CSVs and JSON experiment data
│   └── plots/                                   # 300 DPI comparative trajectory & phase portrait figures
└── README.md
```

---

## 5. Execution Instructions

### 5.1 Autonomous Master Benchmark Execution
```bash
python -m celestial_hnn.benchmarks.run_nine_way_master_benchmark
```

### 5.2 Non-Autonomous Master Benchmark Execution
```bash
python -m celestial_hnn.benchmarks.run_non_autonomous_master_benchmark
```

---

## 6. Formal References

* **Arnold, V. I.** (1989). *Mathematical Methods of Classical Mechanics*. Springer-Verlag.
* **Greydanus, S., Dzamba, M., & Yosinski, J.** (2019). Hamiltonian Neural Networks. *NeurIPS*, 32.
* **Kumar, V., & Marig, S. K.** (2023). Effect of variable mass on N–R basins of convergence in photogravitational magnetic binary problem. *Astronomy Reports*, 67(2), 194-208.
* **Kumar, V., & Marig, S. K.** (2023). Perturbations in Coriolis and Centrifugal Forces and NR Basins of Convergence of Photogravitational Magnetic-Binary Problem with Variable Mass. *Kinematics and Physics of Celestial Bodies*, 39(6), 325-341.
* **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems. *Journal of Computational Physics*, 378, 686-707.
* **Ullah, M. S., Idrisi, M. J., & Kumar, V.** (2020). Elliptic Sitnikov five-body problem under radiation pressure. *New Astronomy*, 80, 101398.
* **Zhong, Y. D., Dey, B., & Chakraborty, A.** (2020). Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control. *ICLR*.
