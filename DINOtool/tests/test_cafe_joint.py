"""Joint gradient, independent teacher, checkpoint, and paired-data contracts."""
import copy
import os
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.getenv("CAFE_OFFICIAL_ROOT", str(ROOT.parent / "third_party/DINO_Soars-f6704c1b76137543328567c8f8b49a2dc318824c")))
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests"), str(OFFICIAL / "CAFe_DINO")]
from modeling.cafedino import CAFe_DINO
from test_cafe_vc import SmallUpsampler, inputs
from dinotool.cafe_joint import JointCafe, JointConfig, joint_loss
from dinotool.cafe_vc import CafeVC, CafeVCConfig
from dinotool.joint_sampling import SeededSourceDataset, SourceBatchSampler, domain_at, consumed_updates
from eval_cafe_vc_protocols import checkpoint_config


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.visual_model = nn.Module()
        self.visual_model.backbone = nn.Module()
        self.visual_model.backbone.blocks = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(4)])
        self.text_model = nn.Linear(8, 8)

    def encode_image_with_patch_tokens(self, images, normalize=False):
        x = self.conv(F.avg_pool2d(images, 16)).flatten(2).transpose(1, 2)
        for block in self.visual_model.backbone.blocks:
            x = x + 0.1 * torch.tanh(block(x))
        return x.mean(1), torch.tanh(x), x


def original():
    return CAFe_DINO(TinyVision(), None, SmallUpsampler(), (7, 7), "cpu", aggregator_dim=16, aggregator_blocks=2).eval()


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(47)
    torch.set_num_threads(2)
    with torch.backends.mkldnn.flags(enabled=False), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        yield


@pytest.mark.parametrize("arm", ["plain", "concat", "rs", "rs_aux"])
def test_joint_training_updates_only_allowed_modules_and_saves_both_blocks(arm, tmp_path):
    base = original()
    teacher = CafeVC(copy.deepcopy(base), CafeVCConfig(arm="plain", blocks=0)).eval().requires_grad_(False)
    config = JointConfig(arm=arm, rs_dim=32, concat_dim=32, stages=2)
    model = JointCafe(copy.deepcopy(base), config, official_root=str(OFFICIAL)).eval()
    image, text = inputs(classes=4)
    target = torch.zeros(1, 112, 112, dtype=torch.long)
    target[:, 48:, :] = 1
    target[:, :16, :16] = 255
    with torch.no_grad():
        reference = teacher(image, text)["logits"]
        before = model(image, text)["logits"]
    torch.testing.assert_close(before, reference, atol=3e-6, rtol=1e-5)
    teacher_state = copy.deepcopy(teacher.state_dict())
    frozen_state = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-3, 1e-3))
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = joint_loss(model(image, text), target, reference, auxiliary=arm == "rs_aux")
        loss.backward()
        optimizer.step()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
            torch.testing.assert_close(p, frozen_state[name], atol=0, rtol=0)
    for i in (2, 3):
        assert not torch.equal(model.cafe.backbone.visual_model.backbone.blocks[i].weight,
                               base.backbone.visual_model.backbone.blocks[i].weight)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_state[name], atol=0, rtol=0)
    assert not any(m.training for m in model.cafe.reduce_d.modules() if isinstance(m, nn.BatchNorm2d))
    state = model.adapted_state_dict()
    assert set(state["visual_blocks"]) == {"2", "3"}
    assert any(name.startswith("cafe.reduce_d.") for name in state["decoder"])
    assert not any(name.startswith(("cafe.backbone.", "cafe.upsampler.")) for name in state["decoder"])
    path = tmp_path / "joint.pt"
    torch.save(dict(format="cafe_joint_v1", architecture=model.architecture(), adapted_state=state), path)
    payload = torch.load(path, weights_only=False)
    assert checkpoint_config(payload) == config
    restored = JointCafe(copy.deepcopy(base), config, official_root=str(OFFICIAL)).eval()
    restored.load_adapted_state_dict(payload["adapted_state"])
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(image, text)["logits"], model(image, text)["logits"], rtol=1e-5, atol=3e-6)
    broken = copy.deepcopy(state)
    del broken["visual_blocks"]["3"]
    with pytest.raises(ValueError):
        restored.load_adapted_state_dict(broken)


def test_warmup_disables_old_module_updates_without_rebuilding_ddp():
    from train_cafe_joint import set_rates
    from argparse import Namespace
    model = JointCafe(original(), JointConfig(arm="concat", concat_dim=32, rs_dim=32, stages=2), official_root=str(OFFICIAL)).train()
    optimizer = torch.optim.AdamW(model.optimizer_groups(1e-3, 1e-4, 1e-5))
    args = Namespace(warmup_steps=2, updates=4)
    enabled = {name for name, p in model.named_parameters() if p.requires_grad}
    set_rates(model, optimizer, args, 0)
    assert all(g["lr"] == 0 for g in optimizer.param_groups if g["role"] != "new")
    assert not model.cafe.aggregator.training
    set_rates(model, optimizer, args, 2)
    assert model.cafe.aggregator.training
    assert all(g["lr"] > 0 for g in optimizer.param_groups)
    assert enabled == {name for name, p in model.named_parameters() if p.requires_grad}


class RandomDataset(Dataset):
    def __len__(self):
        return 11

    def __getitem__(self, index):
        return index, torch.rand(3)


def test_sampling_resume_matches_full_sequence_and_augmentation():
    dataset = SeededSourceDataset(RandomDataset(), 17, 1)
    kwargs = dict(rank=1, world=2, batch_size=2, seed=19)
    full = list(SourceBatchSampler(dataset, **kwargs, start_batch=0, batches=12))
    resumed = list(SourceBatchSampler(dataset, **kwargs, start_batch=5, batches=7))
    assert full[5:] == resumed
    for batch in resumed:
        for key in batch:
            first = dataset[key][1]
            torch.rand(100)
            torch.testing.assert_close(dataset[key][1], first, atol=0, rtol=0)
    for step in range(10):
        assert consumed_updates(step, "coco") + consumed_updates(step, "oem") == step
        assert domain_at(step) == ("coco" if step % 2 == 0 else "oem")


def test_joint_ignore_loss_and_teacher_detach():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    loss, _ = joint_loss({"logits": logits}, torch.full((1, 4, 4), 255), teacher, auxiliary=False)
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert logits.grad is not None and teacher.grad is None


def test_resume_allows_rank_device_but_rejects_protocol_changes():
    from argparse import Namespace
    from train_cafe_joint import validate_resume
    args = Namespace(arm="rs", seed=17, resume="last.pt", output_dir="new")
    base = dict(checkpoint={"path": "official.pt", "bytes": 200}, device="cuda:0")
    payload = dict(format="cafe_joint_v1", architecture={"name": "joint"}, world_size=8,
                   args=vars(args).copy(), source={"source": "locked"}, base_checkpoint=base)
    validate_resume(payload, args, payload["source"], {**base, "device": "cuda:7"}, payload["architecture"], 8)
    with pytest.raises(ValueError, match="provenance"):
        validate_resume(payload, Namespace(**{**vars(args), "seed": 18}), payload["source"], base, payload["architecture"], 8)
