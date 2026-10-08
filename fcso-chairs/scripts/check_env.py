"""Milestone 1 check: MPS is available and both checkpoints load into the repo's models."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from fcso import compat

compat.install()
import models_ae
import models_class_cond

ROOT = os.path.dirname(compat.VECSET_DIR.rstrip("/").rsplit("/", 1)[0])
print("torch", torch.__version__, "mps available:", torch.backends.mps.is_available())

ae = models_ae.kl_d512_m512_l8()
sd = torch.load(os.path.join(ROOT, "weights/ae/kl_d512_m512_l8/checkpoint-199.pth"), map_location="cpu", weights_only=False)
print("ae checkpoint keys:", list(sd.keys()))
print("ae load:", ae.load_state_dict(sd["model"]))
dm = models_class_cond.kl_d512_m512_l8_d24_edm()
sd = torch.load(os.path.join(ROOT, "weights/dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth"), map_location="cpu", weights_only=False)
print("dm checkpoint keys:", list(sd.keys()))
print("dm load:", dm.load_state_dict(sd["model"]))
print("params: ae %.1fM, dm %.1fM" % (sum(p.numel() for p in ae.parameters()) / 1e6, sum(p.numel() for p in dm.parameters()) / 1e6))
