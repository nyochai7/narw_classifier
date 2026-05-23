import torch

from narw_classifier.perch.linear_probe import PerchLinearProbe


class TestPerchLinearProbe:
    def test_forward_produces_1d_logits(self):
        model = PerchLinearProbe(embedding_dim=1536).eval()
        with torch.no_grad():
            logits = model(torch.randn(4, 1536))
        assert logits.shape == (4,)
        assert logits.dtype == torch.float32

    def test_training_step_produces_finite_loss(self):
        model = PerchLinearProbe(embedding_dim=1536).train()
        emb = torch.randn(8, 1536)
        label = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loss = model.training_step((emb, label), batch_idx=0)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_configure_optimizers_is_adamw(self):
        model = PerchLinearProbe(embedding_dim=1536, lr=3e-3, weight_decay=2e-4)
        opt = model.configure_optimizers()
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 3e-3
        assert opt.param_groups[0]["weight_decay"] == 2e-4

    def test_only_linear_layer_in_net(self):
        # Strict linear probe — single Linear, no MLP hidden layer.
        model = PerchLinearProbe(embedding_dim=1536)
        assert isinstance(model.net, torch.nn.Linear)
        assert model.net.in_features == 1536
        assert model.net.out_features == 1
