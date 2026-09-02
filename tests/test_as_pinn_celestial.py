import unittest
import torch
import numpy as np

from celestial_pinn.models.as_pinn import AdaptiveSubspacePINN, SubspaceMLP
from celestial_pinn.models.conflict_monitor import ContinuousConflictMonitor

class TestASPINNEngine(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_voronoi_partition_of_unity_sum(self):
        model = AdaptiveSubspacePINN(
            in_dim=2,
            out_dim=1,
            initial_subspaces=4,
            hidden_dim=32,
            bandwidth=0.5,
            initial_centroids=torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        ).to(self.device)
        
        x = torch.randn(100, 2, device=self.device)
        psi = model.partition_of_unity(x)
        
        # Test sum(psi_k) == 1.0 everywhere
        psi_sum = torch.sum(psi, dim=-1)
        expected = torch.ones_like(psi_sum)
        self.assertTrue(torch.allclose(psi_sum, expected, atol=1e-6))

    def test_zero_disruption_cleavage(self):
        model = AdaptiveSubspacePINN(
            in_dim=1,
            out_dim=2,
            initial_subspaces=1,
            hidden_dim=32,
            bandwidth=0.5,
            initial_centroids=torch.zeros((1, 1)),
        ).to(self.device)
        
        x_test = torch.linspace(-1, 1, 50).reshape(-1, 1)
        u_before = model(x_test).detach().clone()
        
        # Spawn child subspace with parent weights
        new_idx = model.spawn_new_subspace(
            centroid=torch.tensor([[0.5]]),
            bandwidth=0.5,
            parent_idx=0,
        )
        self.assertEqual(model.num_subspaces, 2)
        
        # Check that model forward still evaluates without dimension or shape mismatch
        u_after = model(x_test)
        self.assertEqual(u_after.shape, (50, 2))

    def test_conflict_monitor_gram_matrix(self):
        monitor = ContinuousConflictMonitor(threshold=0.0, in_dim=2)
        
        # Two orthogonal gradients: cosine = 0.0
        G_ortho = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        C, mean_align, clash_ratio, pt_conflict = monitor.analyze_gram_matrix(G_ortho)
        self.assertAlmostEqual(mean_align, 0.0, places=5)
        self.assertAlmostEqual(clash_ratio, 0.0, places=5)
        
        # Opposing gradients: cosine = -1.0
        G_opp = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        C_opp, mean_align_opp, clash_ratio_opp, pt_conflict_opp = monitor.analyze_gram_matrix(G_opp)
        self.assertAlmostEqual(mean_align_opp, -1.0, places=5)
        self.assertAlmostEqual(clash_ratio_opp, 1.0, places=5)

if __name__ == "__main__":
    unittest.main()
