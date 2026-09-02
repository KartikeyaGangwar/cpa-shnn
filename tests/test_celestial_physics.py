import unittest
import torch
import numpy as np

from celestial_pinn.physics.binary_quasar import BinaryQuasarSystem
from celestial_pinn.physics.restricted_six_body import RestrictedSixBodySquareSystem
from celestial_pinn.physics.sitnikov_five_body import EllipticSitnikovFiveBodySystem
from celestial_pinn.physics.magnetic_binary_yukawa import PhotogravitationalMagneticYukawaBinary
from celestial_pinn.solvers.basin_analyzer import BasinEntropyAnalyzer

class TestCelestialPhysics(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        
    def test_binary_quasar_potential_and_gradients(self):
        system = BinaryQuasarSystem(device=self.device)
        x = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([[0.2]], dtype=torch.float32, requires_grad=True)
        
        omega = system.potential(x, y)
        d_omega_dx_auto = torch.autograd.grad(omega, x, create_graph=True)[0]
        d_omega_dy_auto = torch.autograd.grad(omega, y, create_graph=True)[0]
        
        d_omega_dx_ana, d_omega_dy_ana = system.potential_grad(x, y)
        
        self.assertTrue(torch.allclose(d_omega_dx_auto, d_omega_dx_ana, atol=1e-5))
        self.assertTrue(torch.allclose(d_omega_dy_auto, d_omega_dy_ana, atol=1e-5))

    def test_restricted_six_body_symmetry(self):
        system = RestrictedSixBodySquareSystem(device=self.device)
        x_pt = torch.tensor([[0.4]], dtype=torch.float32)
        y_pt = torch.tensor([[0.0]], dtype=torch.float32)
        
        omega_x = system.potential(x_pt, y_pt)
        omega_y = system.potential(y_pt, x_pt)
        self.assertAlmostEqual(omega_x.item(), omega_y.item(), places=5)

    def test_sitnikov_orbital_radius(self):
        system = EllipticSitnikovFiveBodySystem(eccentricity=0.20, radiation_q=1.0, device=self.device)
        v_peri = torch.tensor([[0.0]], dtype=torch.float32)
        r_peri = system.orbital_radius(v_peri)
        self.assertAlmostEqual(r_peri.item(), 0.80, places=5)
        
        v_apo = torch.tensor([[np.pi]], dtype=torch.float32)
        r_apo = system.orbital_radius(v_apo)
        self.assertAlmostEqual(r_apo.item(), 1.20, places=5)

    def test_yukawa_limit_to_newtonian(self):
        # When alpha=0 and q1=q2=1.0, M1=M2=0, Yukawa potential must match standard Newtonian potential exactly
        system_yukawa_zero = PhotogravitationalMagneticYukawaBinary(
            mu=0.35, q1=1.0, q2=1.0, alpha=0.0, M1=0.0, M2=0.0, eps=0.10, device=self.device
        )
        system_quasar = BinaryQuasarSystem(mu=0.35, eps1=0.10, eps2=0.10, device=self.device)
        
        x = torch.tensor([[0.3]], dtype=torch.float32)
        y = torch.tensor([[0.4]], dtype=torch.float32)
        
        omega_y0 = system_yukawa_zero.potential(x, y)
        omega_q = system_quasar.potential(x, y)
        self.assertAlmostEqual(omega_y0.item(), omega_q.item(), places=5)

    def test_basin_entropy_calculation(self):
        uniform_map = np.zeros((20, 20), dtype=np.int32)
        s_b_uniform = BasinEntropyAnalyzer.compute_basin_entropy(uniform_map, box_size=5)
        self.assertAlmostEqual(s_b_uniform, 0.0, places=5)
        
        checker_map = np.indices((20, 20)).sum(axis=0) % 2
        s_b_checker = BasinEntropyAnalyzer.compute_basin_entropy(checker_map, box_size=5)
        self.assertGreater(s_b_checker, 0.5)

if __name__ == "__main__":
    unittest.main()
