"""Pasang KD custom sebelum optimizer/EMA dibuat melalui trainer lokal."""
from pathlib import Path
import yaml
from ultralytics.utils.torch_utils import unwrap_model
from inatrc_kd.pipeline import SafeDetectionTrainer


class CustomKDTrainer(SafeDetectionTrainer):
    def set_model_attributes(self):
        super().set_model_attributes()
        if self.args.compile or self.args.resume:
            raise ValueError("Custom KD pilot does not support compile or resume")
        config = yaml.safe_load((Path(self.save_dir) / "resolved_config.yaml").read_text(encoding="utf-8"))
        # Bungkus model di tahap ini agar adapter KD sudah terdaftar saat optimizer dibuat.
        if config["method"] == "crosskd":
            from .crosskd import CrossKDModel
            self.model = CrossKDModel(self.args.distill_model, self.model, config["kd"]["crosskd"])
        elif config["method"] == "csakd":
            from .csakd import CSAKDModel
            self.model = CSAKDModel(self.args.distill_model, self.model, config["kd"]["csakd"])
        else:
            raise ValueError("CustomKDTrainer requires crosskd or csakd")

    def train(self):
        try:
            return super().train()
        finally:
            # Bersihkan hook juga saat training gagal, supaya fitur lama tidak tertahan.
            model = unwrap_model(self.model)
            if hasattr(model, "_remove_feature_hooks"):
                model._remove_feature_hooks()
                model._student_feats.clear()
                model._teacher_feats.clear()
