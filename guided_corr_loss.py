import torch
from torch import nn
import torch.nn.functional as F


def cosine_dist(A: torch.Tensor, B: torch.Tensor):
    """A, B are [N, C, NP]"""
    A = A - torch.mean(A, dim=2, keepdim=True)
    B = B - torch.mean(B, dim=2, keepdim=True)
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    # dist = 1 - A.transpose(1, 2) @ B
    dist = 1 - (A.transpose(1, 2).to(torch.float16) @ B.to(torch.float16)).to(
        torch.float32
    )
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
    N = tgt_patches.shape[0]
    NP1 = tgt_patches.shape[2]
    NP2 = ref_patches.shape[2]

    dist = cosine_dist(tgt_patches, ref_patches)

    with torch.no_grad():
        n_match = torch.zeros((N, NP2), device=tgt_patches.device, dtype=torch.float32)
        n_match.scatter_add_(
            1,
            torch.argmin(dist, dim=2),
            torch.ones(dist.shape[:2], device=tgt_patches.device, dtype=torch.float32),
        )
        avg_match = NP1 / NP2
        occurrence_penalty = (coef_occur / avg_match) * n_match.unsqueeze(0)

    dist += occurrence_penalty

    return dist


@torch.compile
def guided_corr_loss(
    tgt_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    size: int,
    stride: int,
    coef_occur: float,
    h: float = 0.5,
    eps: float = 1e-5,
    max_patches: int = 8192,
):
    """Calculate the patch guided correspondence loss of reference and target features.
    If the patch size is larger than the feature map size, no loss is calculated and 0 is returned.
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
    if min(*tgt_feat.shape[2:], *ref_feat.shape[2:]) < size:
        return torch.tensor(0.0, device=tgt_feat.device)

    tgt_patches = sample_patches(tgt_feat, size, stride, max_patches)
    ref_patches = sample_patches(ref_feat, size, stride, max_patches)

    dist = guided_corr_dist(tgt_patches, ref_patches, coef_occur)

    w = torch.exp((1 - dist / (torch.amin(dist, dim=2, keepdim=True) + eps)) / h)
    cx = w / torch.sum(w, dim=2, keepdim=True)
    loss = torch.mean(-torch.log(torch.amax(cx, dim=2)))

    return loss
