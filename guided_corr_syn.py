from guided_corr_loss import guided_corr_loss
from net import save_image
from VGG19 import VGG19_AvgPool
import torch
from torch import nn
from torch.nn import functional as F
import contextlib


def synthesize_texture(
    model: nn.Module,
    example: torch.Tensor,
    save_path: str,
    device: torch.device,
    layers,
    save_epoch,
    epochs=2000,
    lr=0.01,
    optimizer="Adam",
    bf16=False,
    target_size: tuple[int] = (256, 256),
    scales=[0.25, 0.5, 0.75, 1.0],
    patch_size=7,
    patch_stride=3,
):
    """Synthesize texture using guided correspondence loss.
    Args:
        example: The example texture image. Shape: [1, 3, H, W] (including batch_size).
    """

    assert optimizer == "Adam", (
        "Only Adam optimizer is supported in guided correspondence synthesis."
    )

    model.to(device)
    example = example.to(device)
    syn = None
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if bf16
        else contextlib.nullcontext()
    )

    for scale in scales:
        # Synthesize texture at different scales. Resample the example and target
        # image accordingly.

        scaled_size = (1, int(target_size[0] * scale), int(target_size[1] * scale))
        example_scaled = F.interpolate(
            example,
            size=scaled_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        if syn is None:
            syn = torch.randn(scaled_size, device=device, requires_grad=True)
        else:
            syn = F.interpolate(
                syn,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).requires_grad_()

        # Calculate example feature maps.
        with torch.no_grad():
            model(example_scaled, layers)
            # feature_maps: list of [N, C, H, W]. Shallow copy the list.
            ref_feats = model.feature_maps.copy()

        for i in range(epochs):
            optimizer.zero_grad()
            with autocast_ctx:
                model(syn, layers)
                tgt_feats = model.feature_maps
                loss = torch.tensor(0.0, device=device)
                for tgt_feat, ref_feat in zip(tgt_feats, ref_feats):
                    loss += guided_corr_loss(
                        tgt_feat,
                        ref_feat,
                        patch_size,
                        patch_stride,
                        coef_occur=0.0,
                    )

            loss.backward()
            optimizer.step()

    save_image(syn.squeeze(0), save_path)
