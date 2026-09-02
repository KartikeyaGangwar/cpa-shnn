# 🌌 Symplectic Hamiltonian Neural Networks (HNN) for Chaotic Celestial Mechanics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BSc Mathematics (Honours) Undergraduate Dissertation Project**  
> *Under the Supervision of Prof. Vinay Kumar*

---

## 📌 Abstract & Mathematical Foundation

This repository provides an exact geometric deep learning framework implementing **Symplectic Hamiltonian Neural Networks (HNN)** (*Greydanus et al., NeurIPS 2019*) to solve complex, chaotic multi-body celestial Hamiltonian dynamical systems.

### ⚖️ Canonical Symplectic Formulation:
Instead of directly fitting empirical vector fields or arbitrary trajectories, the network parameterizes the underlying scalar **Hamiltonian energy manifold** $\mathcal{H}_{\theta}(\mathbf{q}, \mathbf{p}): \mathbb{R}^{2d} \to \mathbb{R}$. The physical equations of motion are generated strictly through the canonical symplectic operator $\mathbf{J}$:

$$\dot{\mathbf{z}} = \begin{pmatrix} \dot{\mathbf{q}} \\ \dot{\mathbf{p}} \end{pmatrix} = \mathbf{J}_{2d} \nabla_{\mathbf{z}} \mathcal{H}_{\theta}(\mathbf{z}) = \begin{pmatrix} 0 & \mathbf{I}_d \\ -\mathbf{I}_d & 0 \end{pmatrix} \begin{pmatrix} \nabla_{\mathbf{q}} \mathcal{H}_{\theta} \\ \nabla_{\mathbf{p}} \mathcal{H}_{\theta} \end{pmatrix}$$

$$\implies \dot{\mathbf{q}} = \frac{\partial \mathcal{H}_{\theta}}{\partial \mathbf{p}}, \quad \dot{\mathbf{p}} = -\frac{\partial \mathcal{H}_{\theta}}{\partial \mathbf{q}}$$

### 🔬 Key Mathematical Properties:
1. **Exact Energy Invariance:** $\frac{d\mathcal{H}}{dt} = (\nabla \mathcal{H})^T \mathbf{J} (\nabla \mathcal{H}) \equiv 0$ identically due to the skew-symmetry $\mathbf{J}^T = -\mathbf{J}$.
2. **Symplecticity (Poincaré Invariance):** Preserves the differential 2-form $\omega = \sum_{i=1}^d dq_i \wedge dp_i$ across all flow maps $\phi_t$.
3. **Liouville's Incompressibility:** $\nabla_{\mathbf{z}} \cdot \dot{\mathbf{z}} = \text{Tr}(\mathbf{J} \nabla^2 \mathcal{H}) \equiv 0$, guaranteeing zero phase space volume dissipation.

---

## 🪐 Benchmark Celestial Systems (Prof. Vinay Kumar Portfolio)

1. **System I: Binary Quasar Hamiltonian System**  
   *Reference:* Kumar et al., *New Astronomy*, 2021 (101543).
2. **System II: Restricted Six-Body Problem with Square Configuration**  
   *Reference:* Kumar et al., *New Astronomy*, 2021 (101451).
3. **System III: Elliptic Sitnikov Five-Body Problem Under Radiation Pressure**  
   *Reference:* Ullah, Idrisi, Kumar, *New Astronomy*, 2020 (101398).
4. **System IV: Photogravitational Magnetic Binary with Non-Newtonian Yukawa Fifth-Force**  
   *Reference:* Kumar, Aggarwal, Marig, *Astronomy and Computing*, 2023 (100783).

---

## ⚡ Quickstart

```bash
# Clone & install dependencies
git clone https://github.com/KartikeyaGangwar/as-pinn-celestial.git
cd as-pinn-celestial
pip install -r requirements.txt

# Run Symplectic Unit Tests
python -m unittest discover -s tests -p "test_*.py"

# Run Master Benchmark Suite
python celestial_hnn/benchmarks/run_master_benchmarks.py
```
