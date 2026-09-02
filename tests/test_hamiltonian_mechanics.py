import unittest
import torch
import numpy as np

from celestial_hnn.models.hnn import HamiltonianNeuralNetwork
from celestial_hnn.physics.binary_quasar import BinaryQuasarHamiltonianSystem
from celestial_hnn.physics.restricted_six_body import RestrictedSixBodyHamiltonianSystem
from celestial_hnn.physics.sitnikov_five_body import SitnikovFiveBodyHamiltonianSystem
from celestial_hnn.physics.magnetic_binary_yukawa import MagneticYukawaHamiltonianSystem

class TestHamiltonianMechanics(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_symplectic_matrix_properties(self):
        hnn = HamiltonianNeuralNetwork(spatial_dim=2).to(self.device)
        J = hnn.J
        # J must be skew-symmetric: J^T = -J
        self.assertTrue(torch.allclose(J.T, -J, atol=1e-7))
        # J^2 = -I
        I = torch.eye(4)
        self.assertTrue(torch.allclose(J @ J, -I, atol=1e-7))

    def test_exact_energy_conservation_property(self):
        hnn = HamiltonianNeuralNetwork(spatial_dim=2).to(self.device)
        z = torch.randn(20, 4, requires_grad=True, device=self.device)
        H = hnn.hamiltonian(z)
        grad_H = torch.autograd.grad(H, z, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        
        # Symplectic flow: dz/dt = (grad_H) @ J^T
        dz_dt = hnn.time_derivative(z)
        
        # dH/dt = grad_H . dz/dt = sum(grad_H * dz_dt) == 0 everywhere!
        dH_dt = torch.sum(grad_H * dz_dt, dim=-1)
        self.assertTrue(torch.allclose(dH_dt, torch.zeros_like(dH_dt), atol=1e-5))

    def test_binary_quasar_hamiltonian_and_derivatives(self):
        system = BinaryQuasarHamiltonianSystem(device=self.device)
        z = torch.tensor([[0.5, 0.2, 0.1, 0.4]], requires_grad=True, device=self.device)
        
        H = system.exact_hamiltonian(z)
        grad_H = torch.autograd.grad(H, z)[0]
        
        dq_dt_auto = grad_H[:, 2:]
        dp_dt_auto = -grad_H[:, :2]
        dz_dt_auto = torch.cat([dq_dt_auto, dp_dt_auto], dim=-1)
        
        dz_dt_ana = system.canonical_derivatives(z)
        self.assertTrue(torch.allclose(dz_dt_auto, dz_dt_ana, atol=1e-5))

    def test_restricted_six_body_symmetry(self):
        system = RestrictedSixBodyHamiltonianSystem(device=self.device)
        # S4 square geometry symmetry at zero momentum
        z1 = torch.tensor([[0.4, 0.0, 0.0, 0.0]], device=self.device)
        z2 = torch.tensor([[0.0, 0.4, 0.0, 0.0]], device=self.device)
        H1 = system.exact_hamiltonian(z1)
        H2 = system.exact_hamiltonian(z2)
        self.assertAlmostEqual(H1.item(), H2.item(), places=5)

    def test_sitnikov_hamiltonian_derivatives(self):
        system = SitnikovFiveBodyHamiltonianSystem(device=self.device)
        z = torch.tensor([[0.5, 0.2]], requires_grad=True, device=self.device)
        H = system.exact_hamiltonian(z)
        grad_H = torch.autograd.grad(H, z)[0]
        dz_dt_auto = torch.cat([grad_H[:, 1:], -grad_H[:, :1]], dim=-1)
        dz_dt_ana = system.canonical_derivatives(z)
        self.assertTrue(torch.allclose(dz_dt_auto, dz_dt_ana, atol=1e-5))

    def test_yukawa_hamiltonian_derivatives(self):
        system = MagneticYukawaHamiltonianSystem(device=self.device)
        z = torch.tensor([[0.4, 0.3, 0.2, 0.1]], requires_grad=True, device=self.device)
        H = system.exact_hamiltonian(z)
        grad_H = torch.autograd.grad(H, z)[0]
        dz_dt_auto = torch.cat([grad_H[:, 2:], -grad_H[:, :2]], dim=-1)
        dz_dt_ana = system.canonical_derivatives(z)
        self.assertTrue(torch.allclose(dz_dt_auto, dz_dt_ana, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
