"""Paired frozen epsilon-loss observations on unchanged Geometry support."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


IMPLEMENTATION = "geometry-supported-paired-epsilon-observer-v1-20261002"
METHODS = ("Geometry", "DenoiseLocal20", "DenoiseGeometry20", "DenoiseGlobal20",
           "DenoiseGeometryCanonical")


@dataclass(frozen=True)
class GenerativeObservationConfig:
    timestep_samples: int = 32
    batch_size: int = 8
    seed: int = 20261002
    prompt_template: str = "an aerial photograph of {label}."
    aliases_per_class: int = 20

    def signature(self):
        return {**asdict(self), "implementation": IMPLEMENTATION,
                "noise": "one common epsilon per timestep across all prompts in a window",
                "timesteps": "uniform midpoint quadrature of the trained DDPM schedule",
                "loss": "unweighted squared epsilon error; average channels/timesteps/aliases",
                "geometry": "unchanged dense original G, masked only by image bounds",
                "scope": "frozen observation diagnostic, not calibrated posterior or final segmentation"}


def timestep_grid(total, samples):
    if not 1 <= samples <= total:
        raise ValueError("Require distinct midpoint timesteps within the trained schedule.")
    return ((torch.arange(samples, dtype=torch.float64)+.5)*total/samples).floor().long()


def window_seed(key, base):
    return (int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")+base) % (2**63-1)


def conditional_noise_error(prediction, epsilon):
    if prediction.ndim != 4 or epsilon.shape not in (prediction.shape, (1, *prediction.shape[1:])):
        raise ValueError("Predicted epsilon and paired noise must have compatible [B,C,H,W] shapes.")
    return (prediction.float()-epsilon.float()).square().mean(1)


def reduce_alias_errors(errors, parents, classes):
    if errors.ndim != 3 or parents.shape != errors.shape[:1]:
        raise ValueError("Require per-alias [A,H,W] error maps and parent indices.")
    if not bool(torch.isfinite(errors).all()) or bool((errors < 0).any()):
        raise ValueError("Compatibility errors must be finite and nonnegative.")
    if set(parents.tolist()) != set(range(classes)):
        raise ValueError("Each target class must have aliases and no out-of-range parent.")
    return torch.stack([errors[parents == index].mean(0) for index in range(classes)])


def supported_errors(error_maps, relation, valid):
    if error_maps.ndim != 2 or relation.shape != (len(error_maps), len(error_maps)) or valid.shape != error_maps.shape[:1]:
        raise ValueError("Require [N,C] errors, [N,N] relation and [N] image-only validity.")
    if not all(bool(torch.isfinite(x).all()) for x in (error_maps, relation)) or bool((relation < 0).any()):
        raise ValueError("Require finite error maps and nonnegative Geometry.")
    weights = relation.float().masked_fill(~valid[None], 0.)
    mass = weights.sum(-1, keepdim=True)
    result = weights @ error_maps.float()/mass.clamp_min(1e-30)
    return torch.where((mass > 0) & valid[:, None], result, error_maps)


class FrozenGenerativeObserver:
    def __init__(self, source, alias_names, parents, class_names, device="cuda",
                 config=GenerativeObservationConfig()):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.source, self.config, self.device = Path(source), config, torch.device(device)
        self.manifest = json.loads((self.source/"generative_source_manifest.json").read_text())
        self.parents = torch.as_tensor(parents, device=self.device, dtype=torch.long)
        self.classes = len(class_names)
        if (len(alias_names) != len(self.parents) or any(int((self.parents == c).sum()) != config.aliases_per_class
                                                     for c in range(self.classes))):
            raise ValueError("The observer must compare exactly the unchanged all20 alias groups.")
        common = dict(local_files_only=True, use_safetensors=True)
        self.vae = AutoencoderKL.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32, **common).to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(source, subfolder="unet", variant="fp16",
                                                        torch_dtype=torch.float16, **common).to(self.device).eval()
        encoder = CLIPTextModel.from_pretrained(source, subfolder="text_encoder", variant="fp16",
                                                dtype=torch.float16, **common).to(self.device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(source, subfolder="tokenizer", local_files_only=True)
        self.scheduler = DDPMScheduler.from_pretrained(source, subfolder="scheduler", local_files_only=True)
        if self.scheduler.config.prediction_type != "epsilon":
            raise ValueError("This pinned observer requires an epsilon-prediction checkpoint.")
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        encoder.requires_grad_(False)
        self.scaling = self.vae.config.scaling_factor
        texts = [config.prompt_template.format(label=name) for name in (*alias_names, *class_names)]
        inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt")
        with torch.inference_mode():
            self.embeddings = torch.cat([encoder(inputs.input_ids[start:start+64].to(self.device))[0]
                                         for start in range(0, len(texts), 64)])
        self.alias_count = len(alias_names)
        del encoder
        self.timesteps = timestep_grid(self.scheduler.config.num_train_timesteps, config.timestep_samples)

    @torch.inference_mode()
    def __call__(self, rgb, relation, valid, sample_key):
        if rgb.shape != (1, 3, 512, 512) or relation.shape != (1024, 1024) or valid.shape != (1024,):
            raise ValueError("The fixed diagnostic expects one native512 window and its original Geometry.")
        if not bool(torch.isfinite(rgb).all()) or float(rgb.min()) < 0 or float(rgb.max()) > 1:
            raise ValueError("Require original real image values in [0,1].")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        latent = self.vae.encode(2*rgb.to(self.device, dtype=torch.float32)-1).latent_dist.mean*self.scaling
        if latent.shape != (1, 4, 64, 64):
            raise ValueError("Unexpected pretrained latent geometry.")
        generator = torch.Generator(device=self.device).manual_seed(window_seed(sample_key, self.config.seed))
        errors = torch.zeros((len(self.embeddings), 64, 64), device=self.device, dtype=torch.float32)
        halves = torch.zeros((2, self.classes, 32, 32), device=self.device, dtype=torch.float32)
        for index, step in enumerate(self.timesteps):
            epsilon = torch.randn(latent.shape, device=self.device, dtype=torch.float32, generator=generator)
            current = self.scheduler.add_noise(latent, epsilon, step.reshape(1).to(self.device)).half()
            current_errors = []
            for start in range(0, len(self.embeddings), self.config.batch_size):
                context = self.embeddings[start:start+self.config.batch_size]
                prediction = self.unet(current.expand(len(context), -1, -1, -1),
                                       step.to(self.device), encoder_hidden_states=context).sample
                current_errors.append(conditional_noise_error(prediction, epsilon))
            current_errors = torch.cat(current_errors)
            errors.add_(current_errors/len(self.timesteps))
            grouped = reduce_alias_errors(current_errors[:self.alias_count], self.parents, self.classes)
            halves[index % 2].add_(F.avg_pool2d(grouped[:, None], 2)[:, 0]/((len(self.timesteps)+1-index % 2)//2))
        all20 = reduce_alias_errors(errors[:self.alias_count], self.parents, self.classes)
        local = F.avg_pool2d(all20[:, None], 2)[:, 0].flatten(1).T
        canonical = F.avg_pool2d(errors[self.alias_count:, None], 2)[:, 0].flatten(1).T
        relation, valid = relation.to(self.device), valid.to(self.device)
        supported = supported_errors(local, relation, valid)
        supported_canonical = supported_errors(canonical, relation, valid)
        global_error = local[valid].mean(0).expand_as(local) if valid.any() else local
        half_support = torch.stack([supported_errors(value.flatten(1).T, relation, valid) for value in halves])
        torch.cuda.synchronize(self.device)
        diagnostics = {"wall_seconds": time.perf_counter()-started,
                       "unet_calls": len(self.timesteps)*((len(self.embeddings)+self.config.batch_size-1)//self.config.batch_size),
                       "unet_examples": len(self.timesteps)*len(self.embeddings),
                       "split_schedule_support_agreement": float((half_support[0].argmin(-1) == half_support[1].argmin(-1))[valid].float().mean()) if valid.any() else 0.,
                       "mean_local_error": float(local.mean()),
                       "mean_class_error_range": float((supported.amax(-1)-supported.amin(-1))[valid].mean()) if valid.any() else 0.}
        scores = {"DenoiseLocal20": -local, "DenoiseGeometry20": -supported,
                  "DenoiseGlobal20": -global_error, "DenoiseGeometryCanonical": -supported_canonical}
        return scores, {"all_prompt_errors": errors, "split_schedule_class_errors": halves}, diagnostics
