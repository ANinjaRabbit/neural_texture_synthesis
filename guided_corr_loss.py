import torch
from torch import nn
import torch.nn.functional as F


def cosine_dist(A: torch.Tensor, B: torch.Tensor):
    """A, B are [N, C, NP]"""
    A = F.normalize(A, dim=2)
    B = F.normalize(B, dim=2)
    dist = 1 - A.transpose(1, 2) @ B
    return dist.clamp(min=0)


def sample_patches(feat: torch.Tensor, size: int, stride: int, max_patches: int):
    """Sample patches from feature map.
    Args:
        feat: [N, C, H, W].
        size: the size of a patch.
        stride: the stride of patches.
        max_patches: the maximum number of patches to sample. If the total number
            of patches exceeds this number, random sampling is performed.
    Returns:
        patches: [N, C * size * size, NP], where NP is the number of sampled patches.
    """
    patches = F.unfold(feat, kernel_size=size, stride=stride)
    if patches.shape[2] > max_patches:
        indices = torch.randperm(patches.shape[2])[:max_patches]
        patches = patches[:, :, indices]
    return patches


def guided_corr_dist(
    tgt_patches: torch.Tensor,
    ref_patches: torch.Tensor,
    size: int,
    stride: int,
    coef_occur: float,
) -> torch.Tensor:
    """Calculate the patch guided correspondence distance of reference and target features.
    Args:
        tgt_patches: [N, C, NP1], the patch features of the target image.
        ref_patches: [N, C, NP2], the patch features of the reference image.
        size: the size of a patch.
        stride: the stride of a patch.
        coef_occ: the coefficient for occurrence penalty.
    Returns:
        The distance matrix of size: [N, NP1, NP2], where NP1 and NP2 are the number
        of patches in the reference and target image.
    """
    # Create patch features of shape [N, C * size * size, NP]
    assert coef_occur == 0, "Occurrence penalty is not yet supported."

    cos_dist = cosine_dist(tgt_patches, ref_patches)

    return cos_dist


def guided_corr_loss(
    tgt_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    size: int,
    stride: int,
    coef_occur: float,
    h: float = 0.5,
    eps: float = 1e-5,
    max_patches: int = 10000,
):
    """Calculate the patch guided correspondence loss of reference and target features.
    Args:
        tgt_feat: [N, C, H, W], the feature map of the target image.
                  Can be the activation of a layer in VGG.
        ref_feat: [N, C, H, W], the feature map of the reference image.
                   Can be the activation of a layer in VGG.
        size: the size of a patch.
        stride: the stride of a patch.
        coef_occur: the coefficient for occurrence penalty.
        h: softmax temperature.
        eps: a small value to avoid numerical issues in division.
    Returns:
        A scalar Tensor of the mean loss of the batch.
    """
    tgt_patches = sample_patches(tgt_feat, size, stride, max_patches)
    ref_patches = sample_patches(ref_feat, size, stride, max_patches)

    dist = guided_corr_dist(tgt_patches, ref_patches, size, stride, coef_occur)

    # dist = guided_corr_dist(ref_feat, tgt_feat, size, stride, coef_occ)
    w = torch.exp((1 - dist / (torch.min(dist, dim=2, keepdim=True) + eps)) / h)
    cx = w / torch.sum(w, dim=2, keepdim=True)
    loss = torch.mean(-torch.log(torch.max(cx, dim=2)))

    return loss
