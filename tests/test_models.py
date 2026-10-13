import unittest
import torch
from src.models.deeponet import SimpleDeepONet, masked_mse


class ModelTests(unittest.TestCase):
    def test_masked_loss_and_case_dependent_trunk(self):
        torch.manual_seed(7)
        model = SimpleDeepONet(branch_in=12, latent=8)
        sdf = torch.rand(2, 12)
        trunk = torch.cat((torch.rand(2, 12, 2), sdf.unsqueeze(-1)), dim=-1)
        prediction = model(sdf, trunk)
        self.assertEqual(prediction.shape, (2, 12))
        mask = torch.ones_like(prediction)
        mask[:, 0] = 0
        target = prediction.detach().clone()
        target[:, 0] = 1e10
        self.assertEqual(masked_mse(prediction, target, mask).item(), 0)
        loss = masked_mse(prediction, torch.zeros_like(prediction), mask)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
