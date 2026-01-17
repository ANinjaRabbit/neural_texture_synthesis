import torch
import torch.nn.functional as F


def cosine_dist(A: torch.Tensor, B: torch.Tensor):
    """Cosine distance between patch features.
    A, B are [N, C, NP]
    """
    A = A - torch.mean(A, dim=2, keepdim=True)
    B = B - torch.mean(B, dim=2, keepdim=True)
    A = F.normalize(A, dim=1)
    B = F.normalize(B, dim=1)
    # 保持float32精度以提高兼容性
    dist = 1 - (A.transpose(1, 2) @ B)
    return dist.clamp(min=0)


def annotation_dist(A: torch.Tensor, B: torch.Tensor):
    """L0 distance for annotation patches (discrete labels).
    A, B are [N, C, NP] where C = k*k (flattened patch of integer labels)
    Returns normalized distance [0, 1]
    """
    # A: [N, C, NP1] -> [N, NP1, C]
    # B: [N, C, NP2] -> [N, NP2, C]
    A_t = A.transpose(1, 2).contiguous()  # [N, NP1, C]
    B_t = B.transpose(1, 2).contiguous()  # [N, NP2, C]
    
    # Broadcast compare: [N, NP1, 1, C] vs [N, 1, NP2, C]
    A_exp = A_t.unsqueeze(2)  # [N, NP1, 1, C]
    B_exp = B_t.unsqueeze(1)  # [N, 1, NP2, C]
    
    # L0 distance: count mismatched pixels
    not_equal = (A_exp != B_exp).float()
    dist = not_equal.sum(dim=-1)  # [N, NP1, NP2]
    
    # Normalize by patch size (C)
    return dist / (A.shape[1] + 1e-8)


def sample_patches(feat: torch.Tensor, size: int, stride: int, max_patches: int):
    """Sample patches from feature map.
    Args:
        feat: [N, C, H, W]
        size: patch size
        stride: sampling stride
        max_patches: maximum patches to sample (random subset if exceeds)
    Returns:
        patches: [N, C*size*size, NP]
    """
    patches = F.unfold(feat, kernel_size=size, stride=stride)
    if patches.shape[2] > max_patches:
        indices = torch.randperm(patches.shape[2])[:max_patches]
        patches = patches[:, :, indices]
    return patches


def guided_corr_dist(
    tgt_patches: torch.Tensor,
    ref_patches: torch.Tensor,
    tgt_label_patches: torch.Tensor = None,
    ref_label_patches: torch.Tensor = None,
    coef_occur: float = 0.05,
    coef_gc: float = 10.0,
) -> torch.Tensor:
    """Calculate guided correspondence distance with optional annotation control.
    
    Args:
        tgt_patches: [N, C_feat, NP1], target feature patches
        ref_patches: [N, C_feat, NP2], reference feature patches
        tgt_label_patches: [N, C_label, NP1], target annotation patches (optional)
        ref_label_patches: [N, C_label, NP2], reference annotation patches (optional)
        coef_occur: occurrence penalty coefficient
        coef_gc: annotation distance weight (lambda_GC in paper)
    
    Returns:
        distance matrix: [N, NP1, NP2]
    """
    N = tgt_patches.shape[0]
    NP1 = tgt_patches.shape[2]
    NP2 = ref_patches.shape[2]

    # 1. VGG feature distance (cosine)
    dist_vgg = cosine_dist(tgt_patches, ref_patches)
    
    # 2. Annotation distance (L0) - if provided
    if tgt_label_patches is not None and ref_label_patches is not None:
        dist_gc = annotation_dist(tgt_label_patches, ref_label_patches)
        dist_combined = dist_vgg + coef_gc * dist_gc
    else:
        dist_combined = dist_vgg

    # 3. Occurrence penalty
    with torch.no_grad():
        # Find nearest neighbors based on combined distance
        nn_indices = torch.argmin(dist_combined, dim=2)  # [N, NP1]
        
        # Count occurrences of each reference patch
        n_match = torch.zeros((N, NP2), device=tgt_patches.device, dtype=torch.float32)
        n_match.scatter_add_(
            1,
            nn_indices,
            torch.ones((N, NP1), device=tgt_patches.device, dtype=torch.float32),
        )
        
        # Normalized occurrence penalty
        avg_match = NP1 / NP2
        occurrence_penalty = (coef_occur / avg_match) * n_match.unsqueeze(0)  # [1, N, NP2] -> [N, NP1, NP2]

    # Final distance
    dist = dist_combined + occurrence_penalty

    return dist


def guided_corr_loss(
    tgt_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    size: int,
    stride: int,
    coef_occur: float = 0.05,
    coef_gc: float = 10.0,
    h: float = 0.5,
    eps: float = 1e-5,
    max_patches: int = 10000,
    tgt_label: torch.Tensor = None,
    ref_label: torch.Tensor = None,
):
    """Guided correspondence loss with optional annotation control.
    
    Args:
        tgt_feat: [N, C_feat, H, W], target VGG features
        ref_feat: [N, C_feat, H, W], reference VGG features
        size: patch size
        stride: patch stride
        coef_occur: occurrence penalty coefficient
        coef_gc: annotation distance weight
        h: temperature for contextual similarity
        eps: small constant for numerical stability
        max_patches: maximum patches to sample
        tgt_label: [N, 1, H_img, W_img], target annotation map (optional)
        ref_label: [N, 1, H_img, W_img], reference annotation map (optional)
    
    Returns:
        loss: scalar tensor
    """
    # Check if feature maps are large enough
    if min(*tgt_feat.shape[2:], *ref_feat.shape[2:]) < size:
        return torch.tensor(0.0, device=tgt_feat.device)

    # 1. Sample feature patches
    tgt_feat_patches = sample_patches(tgt_feat, size, stride, max_patches)
    ref_feat_patches = sample_patches(ref_feat, size, stride, max_patches)

    # 2. Handle annotation maps if provided
    tgt_label_patches = None
    ref_label_patches = None
    
    if tgt_label is not None and ref_label is not None:
        # Resize annotation maps to match feature spatial resolution
        feat_h, feat_w = tgt_feat.shape[2], tgt_feat.shape[3]
        ref_feat_h, ref_feat_w = ref_feat.shape[2], ref_feat.shape[3]
        
        # Use nearest neighbor for integer labels
        tgt_label_resized = F.interpolate(
            tgt_label.float(), size=(feat_h, feat_w), mode='nearest'
        ).long()
        
        ref_label_resized = F.interpolate(
            ref_label.float(), size=(ref_feat_h, ref_feat_w), mode='nearest'
        ).long()

        # Sample annotation patches
        tgt_label_patches = sample_patches(tgt_label_resized.float(), size, stride, max_patches)
        ref_label_patches = sample_patches(ref_label_resized.float(), size, stride, max_patches)

    # 4. Compute guided distance
    dist = guided_corr_dist(
        tgt_feat_patches,
        ref_feat_patches,
        tgt_label_patches,
        ref_label_patches,
        coef_occur,
        coef_gc,
    )

    # 5. Contextual similarity conversion
    min_dist = torch.amin(dist, dim=2, keepdim=True) + eps
    w = torch.exp((1 - dist / min_dist) / h)
    cx = w / torch.sum(w, dim=2, keepdim=True)
    
    # 6. Loss: negative log of max contextual similarity
    loss = torch.mean(-torch.log(torch.amax(cx, dim=2)))

    return loss