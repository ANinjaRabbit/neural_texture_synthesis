import torch
import torch.nn.functional as F
import contextlib
import cv2
import numpy as np
from tqdm import tqdm


def read_annotation(path: str, device: torch.device):
    """Read annotation map as integer labels.
    Args:
        path: path to annotation image (grayscale PNG)
    Returns:
        tensor: [1, 1, H, W] with integer labels
    """
    # Read as grayscale (single channel)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Annotation not found: {path}")
    
    # Convert to tensor, keep as integers
    img_tensor = torch.from_numpy(img).long().unsqueeze(0)  # [1, H, W]
    img_tensor = img_tensor.unsqueeze(0)  # [1, 1, H, W]
    
    return img_tensor.to(device)


def augment_source_safe(source_img: torch.Tensor, source_label: torch.Tensor):
    """Safe augmentation with only flips (no rotations) to avoid size mismatch."""
    aug_imgs = []
    aug_labels = []
    
    # Original
    aug_imgs.append(source_img)
    aug_labels.append(source_label)
    
    # Horizontal flip
    aug_imgs.append(torch.flip(source_img, dims=[3]))
    aug_labels.append(torch.flip(source_label, dims=[3]))
    
    # Vertical flip
    aug_imgs.append(torch.flip(source_img, dims=[2]))
    aug_labels.append(torch.flip(source_label, dims=[2]))
    
    # Both flips (horizontal + vertical)
    aug_imgs.append(torch.flip(source_img, dims=[2, 3]))
    aug_labels.append(torch.flip(source_label, dims=[2, 3]))
    
    return torch.cat(aug_imgs, dim=0), torch.cat(aug_labels, dim=0)


def synthesize_texture_gc(
    model,
    gt,
    save_path,
    device,
    layers,
    save_epoch,
    epochs=500,
    lr=0.01,
    optimizer="Adam",
    bf16=False,
    target_size=(256, 256),
    scales=[0.25, 0.5, 0.75, 1.0],
    patch_size=7,
    patch_stride=3,
    source_annot=None,
    target_annot=None,
    coef_gc=10.0,
    augment=True,
):
    """Texture synthesis using Guided Correspondence loss.
    
    Args:
        gt: [1, 3, H, W], source texture image
        source_annot: [1, 1, H, W], source annotation map (optional)
        target_annot: [1, 1, H_t, W_t], target annotation map (optional)
    """
    from guided_corr_loss import guided_corr_loss
    from net import save_image
    
    model.to(device)
    gt = gt.to(device)
    
    # Check if controlled synthesis
    controlled = source_annot is not None and target_annot is not None
    
    if controlled:
        print("Controlled synthesis with annotation maps")
        source_annot = source_annot.to(device)
        target_annot = target_annot.to(device)
        
        # Apply augmentation to source
        if augment:
            gt_aug, source_annot_aug = augment_source_safe(gt, source_annot)
            print(f"Applied safe augmentation: {gt_aug.shape[0]} variants")
        else:
            gt_aug, source_annot_aug = gt, source_annot
    else:
        print("Uncontrolled synthesis")
        gt_aug = gt
    
    syn = None
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if bf16
        else contextlib.nullcontext()
    )
    
    for scale_idx, scale in enumerate(scales):
        # Scale sizes
        target_scaled_size = (int(target_size[0] * scale), int(target_size[1] * scale))
        
        # For controlled synthesis,我们需要考虑源图像和标注图的尺寸
        if controlled:
            # 源图像尺寸
            gt_h, gt_w = gt.shape[2], gt.shape[3]
            gt_scaled_size = (int(gt_h * scale), int(gt_w * scale))
        else:
            gt_scaled_size = (int(gt.shape[2] * scale), int(gt.shape[3] * scale))
        
        # Resize source image (all augmented variants)
        gt_scaled = F.interpolate(
            gt_aug,
            size=gt_scaled_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        
        # Initialize or upsample synthesis
        if syn is None:
            syn = torch.randn(
                (1, 3) + target_scaled_size,
                device=device,
                requires_grad=True,
            )
        else:
            syn = F.interpolate(
                syn,
                size=target_scaled_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).detach_().requires_grad_()
        
        # Setup optimizer
        if optimizer == "Adam":
            optimizer_obj = torch.optim.Adam([syn], lr=lr)
        elif optimizer == "LBFGS":
            from Optimizer import SimpleLBFGS
            optimizer_obj = SimpleLBFGS([syn], lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Extract reference features
        with torch.no_grad():
            if controlled:
                # For controlled: process all augmented variants
                all_ref_feats = []
                for i in range(gt_scaled.shape[0]):
                    single_gt = gt_scaled[i:i+1]
                    model(single_gt, layers)
                    all_ref_feats.append(model.feature_maps.copy())
            else:
                # For uncontrolled: just one set
                model(gt_scaled, layers)
                ref_feats = model.feature_maps.copy()
        
        print(f"[Scale {scale}] Source: {gt_scaled.shape[2:]} -> Target: {target_scaled_size}")
        
        # Optimization loop
        pbar = tqdm(range(epochs), desc=f"Scale {scale}")
        for epoch in pbar:
            if optimizer == "LBFGS":
                def closure():
                    optimizer_obj.zero_grad()
                    with autocast_ctx:
                        model(syn, layers)
                        tgt_feats = model.feature_maps
                        
                        loss = torch.tensor(0.0, device=device)
                        
                        if controlled:
                            # Resize target annotation for this scale
                            target_annot_scaled = F.interpolate(
                                target_annot.float(),
                                size=target_scaled_size,
                                mode='nearest'
                            ).long()
                            
                            # Sum losses over all augmented reference variants
                            for layer_idx in range(len(tgt_feats)):
                                tgt_feat = tgt_feats[layer_idx]
                                layer_loss = torch.tensor(0.0, device=device)
                                
                                for aug_idx in range(gt_scaled.shape[0]):
                                    ref_feat = all_ref_feats[aug_idx][layer_idx]
                                    ref_annot = source_annot_aug[aug_idx:aug_idx+1]
                                    
                                    # Resize source annotation
                                    ref_annot_scaled = F.interpolate(
                                        ref_annot.float(),
                                        size=gt_scaled_size,
                                        mode='nearest'
                                    ).long()
                                    
                                    layer_loss += guided_corr_loss(
                                        tgt_feat,
                                        ref_feat,
                                        size=patch_size,
                                        stride=patch_stride,
                                        coef_occur=0.05,
                                        coef_gc=coef_gc,
                                        h=1.0,
                                        tgt_label=target_annot_scaled,
                                        ref_label=ref_annot_scaled,
                                    )
                                
                                loss += layer_loss / gt_scaled.shape[0]  # Average over augmentations
                        else:
                            # Uncontrolled synthesis
                            for tgt_feat, ref_feat in zip(tgt_feats, ref_feats):
                                loss += guided_corr_loss(
                                    tgt_feat,
                                    ref_feat,
                                    size=patch_size,
                                    stride=patch_stride,
                                    coef_occur=0.05,
                                    coef_gc=0.0,  # No annotation weight
                                    h=1.0,
                                )
                    
                    loss.backward()
                    return loss.detach()
                
                loss = optimizer_obj.step(closure)
            else:
                # Adam optimizer
                optimizer_obj.zero_grad()
                with autocast_ctx:
                    model(syn, layers)
                    tgt_feats = model.feature_maps
                    
                    loss = torch.tensor(0.0, device=device)
                    
                    if controlled:
                        # Resize target annotation for this scale
                        target_annot_scaled = F.interpolate(
                            target_annot.float(),
                            size=target_scaled_size,
                            mode='nearest'
                        ).long()
                        
                        # Sum losses over all augmented reference variants
                        for layer_idx in range(len(tgt_feats)):
                            tgt_feat = tgt_feats[layer_idx]
                            layer_loss = torch.tensor(0.0, device=device)
                            
                            for aug_idx in range(gt_scaled.shape[0]):
                                ref_feat = all_ref_feats[aug_idx][layer_idx]
                                ref_annot = source_annot_aug[aug_idx:aug_idx+1]
                                
                                # Resize source annotation
                                ref_annot_scaled = F.interpolate(
                                    ref_annot.float(),
                                    size=gt_scaled_size,
                                    mode='nearest'
                                ).long()
                                
                                layer_loss += guided_corr_loss(
                                    tgt_feat,
                                    ref_feat,
                                    size=patch_size,
                                    stride=patch_stride,
                                    coef_occur=0.05,
                                    coef_gc=coef_gc,
                                    h=1.0,
                                    tgt_label=target_annot_scaled,
                                    ref_label=ref_annot_scaled,
                                )
                            
                            loss += layer_loss / gt_scaled.shape[0]  # Average over augmentations
                    else:
                        # Uncontrolled synthesis
                        for tgt_feat, ref_feat in zip(tgt_feats, ref_feats):
                            loss += guided_corr_loss(
                                tgt_feat,
                                ref_feat,
                                size=patch_size,
                                stride=patch_stride,
                                coef_occur=0.05,
                                coef_gc=0.0,  # No annotation weight
                                h=1.0,
                            )
                
                loss.backward()
                optimizer_obj.step()
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Save intermediate results
            if save_epoch != 0 and (epoch + 1) % save_epoch == 0:
                save_image(syn.squeeze(0), f"images/epochs/epoch_{scale_idx}_{epoch+1}.jpg")
    
    # Save final result
    save_image(syn.squeeze(0), save_path)
    return syn