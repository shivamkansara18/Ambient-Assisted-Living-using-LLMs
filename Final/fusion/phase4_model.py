"""
Phase 4 — Temporal Attention Fusion Model
==========================================
Implements the cross-attention transformer fusion module.

Architecture
------------
Input:  img_seq [B, W, D]  — window of projected image features
        sen_seq [B, W, D]  — window of projected sensor features

Step 1: Add temporal positional encoding to both streams.
Step 2: Bidirectional cross-attention.
          - Image tokens query sensor tokens  → enriched image context
          - Sensor tokens query image tokens  → enriched sensor context
        Both paths use residual + LayerNorm.
Step 3: Mean-pool each enriched stream over the time dimension → [B, D].
Step 4: Concatenate → [B, 2D], pass through MLP classifier → [B, NUM_CLASSES].

Training-time modality dropout:
    With probability p_drop, one or both modalities are randomly zeroed.
    This forces the model to be robust when a modality is unavailable.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase1_ontology import NUM_UNIFIED, PROJ_DIM


class TemporalFusionTransformer(nn.Module):
    """
    Cross-attention temporal fusion transformer.

    Args:
        feat_dim    : dimension of projected features from each encoder (PROJ_DIM)
        n_heads     : number of attention heads (feat_dim must be divisible by n_heads)
        window      : temporal window size W
        n_classes   : number of unified output classes
        dropout     : dropout probability inside attention and MLP
        p_modal_drop: probability of zeroing one modality during training
    """

    def __init__(
        self,
        feat_dim: int = PROJ_DIM,
        n_heads: int = 4,
        window: int = 8,
        n_classes: int = NUM_UNIFIED,
        dropout: float = 0.2,
        p_modal_drop: float = 0.2,
    ):
        super().__init__()
        assert feat_dim % n_heads == 0, \
            f"feat_dim ({feat_dim}) must be divisible by n_heads ({n_heads})"

        self.feat_dim     = feat_dim
        self.window       = window
        self.p_modal_drop = p_modal_drop

        # ── Positional encoding (learnable) ──────────────────────────────────
        self.pos_emb = nn.Embedding(window, feat_dim)

        # ── Cross-attention layers ────────────────────────────────────────────
        # Image → Sensor attention: image queries, sensor keys/values
        self.img2sen = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Sensor → Image attention: sensor queries, image keys/values
        self.sen2img = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ── Post-attention layer norms (Pre-LN residual style) ────────────────
        self.norm_img_pre  = nn.LayerNorm(feat_dim)
        self.norm_sen_pre  = nn.LayerNorm(feat_dim)
        self.norm_img_post = nn.LayerNorm(feat_dim)
        self.norm_sen_post = nn.LayerNorm(feat_dim)

        # ── Feed-forward after attention (per token) ──────────────────────────
        self.ffn_img = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        self.ffn_sen = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        # Input: concat of pooled img and sen streams → 2 * feat_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim * 2),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _apply_modality_dropout(self, img_seq, sen_seq):
        """
        Randomly zero out one modality with probability p_modal_drop.
        Called only during training.
        """
        if not self.training:
            return img_seq, sen_seq

        r = random.random()
        if r < self.p_modal_drop:
            img_seq = torch.zeros_like(img_seq)
        elif r < 2 * self.p_modal_drop:
            sen_seq = torch.zeros_like(sen_seq)
        return img_seq, sen_seq

    def forward(
        self,
        img_seq: torch.Tensor,   # [B, W, D]
        sen_seq: torch.Tensor,   # [B, W, D]
        force_img_zero: bool = False,
        force_sen_zero: bool = False,
    ) -> torch.Tensor:           # [B, n_classes]

        B, W, D = img_seq.shape
        pos = self.pos_emb(torch.arange(W, device=img_seq.device))  # [W, D]

        # Optional forced zeroing for inference-time robustness testing
        if force_img_zero:
            img_seq = torch.zeros_like(img_seq)
        if force_sen_zero:
            sen_seq = torch.zeros_like(sen_seq)

        # Modality dropout (training only)
        img_seq, sen_seq = self._apply_modality_dropout(img_seq, sen_seq)

        # Add positional encoding
        img_tok = img_seq + pos.unsqueeze(0)   # [B, W, D]
        sen_tok = sen_seq + pos.unsqueeze(0)   # [B, W, D]

        # ── Cross-attention block (Pre-LN) ────────────────────────────────────
        # Image tokens attend to sensor tokens
        img_normed = self.norm_img_pre(img_tok)
        sen_normed = self.norm_sen_pre(sen_tok)

        img_attn, _ = self.img2sen(img_normed, sen_normed, sen_normed)
        sen_attn, _ = self.sen2img(sen_normed, img_normed, img_normed)

        img_ctx = img_tok + img_attn    # residual
        sen_ctx = sen_tok + sen_attn    # residual

        # ── Feed-forward refinement per token ────────────────────────────────
        img_ctx = self.norm_img_post(img_ctx)
        sen_ctx = self.norm_sen_post(sen_ctx)

        img_ctx = img_ctx + self.ffn_img(img_ctx)  # residual
        sen_ctx = sen_ctx + self.ffn_sen(sen_ctx)  # residual

        # ── Temporal mean pooling → [B, D] ────────────────────────────────────
        img_pool = img_ctx.mean(dim=1)
        sen_pool = sen_ctx.mean(dim=1)

        # ── Concat and classify ───────────────────────────────────────────────
        fused = torch.cat([img_pool, sen_pool], dim=-1)  # [B, 2D]
        return self.classifier(fused)                    # [B, n_classes]

    def predict_proba(self, img_seq, sen_seq):
        """Softmax probabilities — convenience method for inference."""
        self.eval()
        with torch.no_grad():
            logits = self(img_seq, sen_seq)
        return F.softmax(logits, dim=-1)


# ── Model summary ─────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, W, D = 8, 8, 256
    model = TemporalFusionTransformer(
        feat_dim=D, n_heads=4, window=W,
        n_classes=NUM_UNIFIED, dropout=0.2,
    )

    img_seq = torch.randn(B, W, D)
    sen_seq = torch.randn(B, W, D)

    model.train()
    logits_train = model(img_seq, sen_seq)
    assert logits_train.shape == (B, NUM_UNIFIED), \
        f"Wrong output shape: {logits_train.shape}"

    model.eval()
    logits_eval = model(img_seq, sen_seq)
    assert logits_eval.shape == (B, NUM_UNIFIED)

    # Test force-zero paths
    logits_imgonly = model(img_seq, sen_seq, force_sen_zero=True)
    logits_senonly = model(img_seq, sen_seq, force_img_zero=True)

    total, trainable = count_parameters(model)
    print(f"Model parameters: {total:,} total / {trainable:,} trainable")
    print(f"Output shape      : {logits_eval.shape}")
    print(f"Image-only output : {logits_imgonly.shape}")
    print(f"Sensor-only output: {logits_senonly.shape}")
    print("Phase 4 model smoke test passed.")