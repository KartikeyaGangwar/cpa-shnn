# Autonomous Adaptive $N$-Subspace PINN (AS-PINN) for Chaotic Celestial Mechanics

Official implementation of **Adaptive $N$-Subspace Physics-Informed Neural Networks (AS-PINN)** applied to chaotic multi-body celestial mechanics, high-frequency parametric resonance, and fractal phase-space basin boundaries.

---

## 🌌 Target Celestial Mechanics Systems

1. **System I: Binary Quasar Model (Fractal Basins of Attraction)**
   * *Reference:* Kumar, V., Aggarwal, R., Sharma, P., Kaur, B. (*New Astronomy*, 2021: 101543).
   * Two massive quasar primaries with Plummer-softened core gravitational potentials in a rotating frame.

2. **System II: Restricted Six-Body Problem with Square Configuration**
   * *Reference:* Kumar, V., Idrisi, M. J., Ullah, M. S. (*New Astronomy*, 2021: 101451).
   * 4 Primaries on a square configuration + 1 central primary + 1 infinitesimal 6th mass. Resolves 5-center localized stiffness and unpredictable Wada basins.

3. **System III: Elliptic Sitnikov Five-Body Problem Under Radiation Pressure**
   * *Reference:* Ullah, M. S., Idrisi, M. J., Kumar, V. (*New Astronomy*, 2020: 101398).
   * 4 Coplanar primaries on elliptic orbits ($e \in [0, 0.5]$) with photogravitational radiation pressure ($q \in (0, 1]$), governing non-autonomous 2nd-order true anomaly oscillations on the $z$-axis.

4. **System IV: Photogravitational Magnetic Binary with Yukawa Fifth-Force Correction**
   * *Reference:* Kumar, V., Aggarwal, R., Marig, S. K. (*Astronomy and Computing*, 2023: 100783); Kumar, V., Arif, M., Ullah, M. S. (*New Astronomy*, 2021: 101475).
   * Multi-physics benchmark: Coupled Newtonian Gravity + Radiation Pressure ($q_1, q_2$) + Magnetic Dipoles ($1/r^3$) + Non-Newtonian Yukawa potential ($V_Y(r) = -\frac{GM}{r}(1+\alpha e^{-r/\lambda})$).

---

## 🌟 Key Innovations

1. **Vectorized Intra-Subspace Gradient Conflict Profiling:**
   Uses PyTorch's native `torch.func.vmap` and `torch.func.grad` to compute per-point parameter gradients and evaluate cosine similarity Gram matrices in real-time.
2. **Autonomous Zero-Disruption Parameter Cleavage:**
   Pinpoints spatial centroids of high gradient conflict and spawns localized Voronoi parameter subspaces ($\Phi_k$) with exact solution invariance upon fission.
3. **Two-Stage Discover-and-Deploy Pipeline:**
   * **Stage 1 (AMR Discovery):** Probe model discovers optimal minimal subspace count $N^*$ and centroids $\{\mathbf{c}_k\}$ through physical quiescence.
   * **Stage 2 (Production Deployment):** Clean $N^*$-subspace model is trained with synchronized AdamW and polished with Full-Horizon Global L-BFGS (Strong-Wolfe).

---

## 🛠️ Repository Architecture

```
UG_R_proj/
├── celestial_pinn/
│   ├── models/
│   │   ├── as_pinn.py              # AdaptiveSubspacePINN core architecture & Voronoi PoU
│   │   └── conflict_monitor.py     # torch.func.vmap Gram conflict analyzer
│   ├── physics/
│   │   ├── base_celestial.py       # Celestial Base Class (ODEs, Basins, Energy Conservations)
│   │   ├── binary_quasar.py        # System I: Binary Quasar Model
│   │   ├── restricted_six_body.py  # System II: Restricted 6-Body Square System
│   │   ├── sitnikov_five_body.py   # System III: Elliptic Sitnikov Five-Body Problem
│   │   └── magnetic_binary_yukawa.py # System IV: Magnetic Binary + Yukawa Correction
│   ├── solvers/
│   │   ├── numerical_reference.py  # High-precision DOP853 & Newton-Raphson solvers
│   │   └── basin_analyzer.py       # Basin entropy, unpredictability & fractal analysis
│   ├── training/
│   │   ├── two_stage_trainer.py    # Stage 1 Discovery & Stage 2 Production Engine
│   │   └── baseline_trainers.py    # Standard PINN, PCGrad, CAGrad baselines
│   └── benchmarks/
│       ├── run_system1_quasar.py
│       ├── run_system2_sixbody.py
│       ├── run_system3_sitnikov.py
│       ├── run_system4_yukawa.py
│       ├── run_master_benchmarks.py
│       └── visualize_celestial_results.py
├── results/
│   ├── data/                       # CSV/JSON convergence & benchmark matrices
│   └── plots/                      # High-resolution phase portraits and field maps
├── tests/
│   ├── test_celestial_physics.py   # Physics potential & gradient unit tests
│   └── test_as_pinn_celestial.py   # Model cleavage & Gram matrix unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Generate High-Resolution Visualizations
```bash
python celestial_pinn/benchmarks/visualize_celestial_results.py
```

### 4. Run Individual or Master Benchmarks
```bash
# Run Sitnikov Five-Body Benchmark
python celestial_pinn/benchmarks/run_system3_sitnikov.py

# Run Full 4-Model Master Benchmark
python celestial_pinn/benchmarks/run_master_benchmarks.py
```

---

## 📄 License
This repository is open-sourced under the MIT License.
