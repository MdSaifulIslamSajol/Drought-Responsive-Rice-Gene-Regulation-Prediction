# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hybrid Architecture for Multi-Variety Rice Classification (Global Motif Counts)
================================================================================
Version 2: Optimized with HierarchicalFocalLoss

Based on comprehensive focal loss comparison results:
- HierarchicalFocalLoss consistently performed best across most model architectures
- Best combination: HybridGatedFusion + HierarchicalFocal (Test AUC: 0.7850)

Key Improvements over v1:
1. HierarchicalFocalLoss replaces standard BCELoss
2. Condition-aware weighting (variety, tissue, stage)
3. Focal loss for hard example mining
4. Class distribution analysis and automatic alpha calculation

Architecture Design:
- Motif features are treated as a 1D sequence (order of motifs in the feature vector)
- Sequential models (LSTM/Transformer) process the motif sequence
- Embedding branches for Variety/Tissue/Stage conditions
- Fusion layer combines both for final Gene Regulation prediction

Usage:
    First run: python rice_motif_pipeline_global.py
    Then run:  python rice_hybrid_classifier_global_v2.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================
import sys
sys.stdout = open("console_output_hybrid_classifier_global_v2", "w")

DATA_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/output_global_data/"
OUTPUT_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/output_global/hybrid_models_v2_hierarchical_focal/"

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
PATIENCE = 25
HIDDEN_DIM = 128          
DROPOUT = 0.4
WEIGHT_DECAY = 0.01

# HierarchicalFocalLoss parameters (optimized from experiments)
FOCAL_GAMMA = 2.0
CONDITION_STRENGTH = 0.3

# Motif names (must match the pipeline)
MOTIF_NAMES = [
    'ABRE', 'ABRE_core', 'CE1', 'CE3', 'DRE_CRT', 'DRE_full', 'LTRE',
    'MYB_core', 'MYB1AT', 'MYB2', 'MYBCORE', 'MYC', 'WBOX', 'WBOX_core',
    'GT1', 'GT1_core', 'HSE', 'HSE_core', 'GCC_box', 'ERE', 'GBOX', 'IBOX',
    'CAAT_box', 'STRE', 'LTRE_like', 'JASE', 'TBOX', 'NAC_core', 'NACRS',
    'ACGT_core', 'ABRE_like', 'AuxRE', 'AuxRE_rev', 'ROOTMOTIF', 'AS1',
    'TATA_box', 'CACTFTPPCA1', 'ICE_box', 'CBF_DRE', 'ACGT', 'CACG', 'CATG',
    'CGTG', 'GCGT', 'AACA', 'CAAC', 'AAAG', 'GATA', 'TATA', 'CAAT', 'GCCG',
    'CGCG', 'TGAC', 'GTAC'
]

# Sequence statistics features
SEQ_STATS = ['length', 'gc_content', 'at_content', 'gc_skew']


# =============================================================================
# HIERARCHICAL FOCAL LOSS (BEST PERFORMING)
# =============================================================================

class HierarchicalFocalLoss(nn.Module):
    """
    Focal Loss with hierarchical weighting for biological conditions.
    
    This loss function was found to be the best performing across most model
    architectures in the comprehensive focal loss comparison experiment.
    
    Loss = -alpha_t * (1 - p_t)^gamma * condition_weight * log(p_t)
    
    Key features:
    - Alpha weighting for class imbalance (auto-calculated from label distribution)
    - Gamma-based focal weighting to focus on hard examples
    - Condition weighting based on variety/tissue/stage rarity
    - Geometric mean of condition weights for balanced contribution
    
    Best Results:
    - HybridGatedFusion + HierarchicalFocal: Test AUC 0.7850, F1 0.7295
    - HybridLSTM + HierarchicalFocal: Test AUC 0.7771, F1 0.7190
    - HybridCNNAttention + HierarchicalFocal: Test AUC 0.7694, F1 0.7106
    """
    
    def __init__(self, variety_weights, tissue_weights, stage_weights,
                 alpha=0.25, gamma=2.0, condition_strength=0.3):
        """
        Args:
            variety_weights: Inverse frequency weights for varieties
            tissue_weights: Inverse frequency weights for tissues
            stage_weights: Inverse frequency weights for stages
            alpha: Weight for positive class (auto-calculated if None)
            gamma: Focal loss focusing parameter (higher = more focus on hard examples)
            condition_strength: Scaling factor for condition weighting (0-1)
        """
        super().__init__()
        
        # Register buffers for condition weights (will be moved to GPU automatically)
        self.register_buffer('variety_weights', torch.FloatTensor(variety_weights))
        self.register_buffer('tissue_weights', torch.FloatTensor(tissue_weights))
        self.register_buffer('stage_weights', torch.FloatTensor(stage_weights))
        
        self.alpha = alpha
        self.gamma = gamma
        self.condition_strength = condition_strength
        self.name = "HierarchicalFocal"
    
    def forward(self, pred, target, variety_onehot, tissue_onehot, stage_onehot):
        """
        Compute the hierarchical focal loss.
        
        Args:
            pred: Predicted probabilities (after sigmoid)
            target: Ground truth labels (0 or 1)
            variety_onehot: One-hot encoded variety
            tissue_onehot: One-hot encoded tissue
            stage_onehot: One-hot encoded stage
            
        Returns:
            Scalar loss value
        """
        # Numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy (element-wise)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal weight: focus more on hard examples
        # p_t = p if y=1, else 1-p
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Class balance weight
        alpha_weight = torch.where(
            target == 1,
            torch.tensor(self.alpha, device=target.device),
            torch.tensor(1 - self.alpha, device=target.device)
        )
        
        # Condition weights (higher for rare conditions)
        v_weight = (variety_onehot * self.variety_weights).sum(dim=1)
        t_weight = (tissue_onehot * self.tissue_weights).sum(dim=1)
        s_weight = (stage_onehot * self.stage_weights).sum(dim=1)
        
        # Geometric mean of condition weights for balanced contribution
        condition_weight = (v_weight * t_weight * s_weight) ** (1/3)
        
        # Scale condition weight influence
        condition_weight = 1 + self.condition_strength * (condition_weight - 1)
        
        # Combine all weights
        total_weight = alpha_weight * focal_weight * condition_weight
        
        return (bce * total_weight).mean()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_condition_weights(df, condition_cols):
    """
    Compute inverse frequency weights for one-hot encoded conditions.
    
    Rare conditions get higher weights to help the model learn them better.
    """
    counts = df[condition_cols].sum(axis=0).values
    weights = len(df) / (len(condition_cols) * counts)
    weights = weights / weights.mean()  # Normalize to mean=1
    return weights


def compute_alpha_from_labels(train_df):
    """
    Compute optimal alpha for focal loss based on label distribution.
    Alpha = proportion of negative class (to upweight positive class if minority)
    """
    label_counts = train_df['label'].value_counts()
    n_neg = label_counts.get(0, 0)
    n_pos = label_counts.get(1, 0)
    alpha = n_neg / (n_neg + n_pos)  # Weight for positive class
    return alpha


def print_class_distribution(df, variety_cols, tissue_cols, stage_cols):
    """Print detailed class distribution information."""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Label distribution
    label_counts = df['label'].value_counts()
    print(f"\nLabel Distribution:")
    print(f"  Down (0): {label_counts.get(0, 0)} ({100*label_counts.get(0, 0)/len(df):.1f}%)")
    print(f"  Up (1):   {label_counts.get(1, 0)} ({100*label_counts.get(1, 0)/len(df):.1f}%)")
    print(f"  Imbalance ratio: {max(label_counts)/min(label_counts):.2f}:1")
    
    # Variety distribution
    print(f"\nVariety Distribution:")
    for col in variety_cols:
        count = df[col].sum()
        name = col.replace('Variety_', '')
        print(f"  {name}: {int(count)} ({100*count/len(df):.1f}%)")
    
    # Tissue distribution
    print(f"\nTissue Distribution:")
    for col in tissue_cols:
        count = df[col].sum()
        name = col.replace('Tissue_', '')
        print(f"  {name}: {int(count)} ({100*count/len(df):.1f}%)")
    
    # Stage distribution
    print(f"\nStage Distribution:")
    for col in stage_cols:
        count = df[col].sum()
        name = col.replace('Stage_', '')
        print(f"  {name}: {int(count)} ({100*count/len(df):.1f}%)")
    
    print("=" * 60 + "\n")


# =============================================================================
# DATASET
# =============================================================================

class HybridDatasetGlobal(Dataset):
    """
    Dataset for global motif counts (no binning).
    
    Features:
    - Motif counts: 1D vector of shape (num_motifs,)
    - Sequence statistics: (4,)
    - Variety/Tissue/Stage: one-hot encoded
    """
    
    def __init__(self, df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols,
                 scaler=None, fit_scaler=False):
        """
        Args:
            df: DataFrame with features and label
            motif_cols: List of motif count columns (e.g., 'ABRE', 'DRE_CRT', etc.)
            seq_stat_cols: ['length', 'gc_content', 'at_content', 'gc_skew']
            variety_cols: One-hot encoded variety columns
            tissue_cols: One-hot encoded tissue columns
            stage_cols: One-hot encoded stage columns
        """
        # Motif features: simple 1D vector
        self.motif_data = df[motif_cols].values.astype(np.float32)
        
        # Sequence statistics
        self.seq_stats = df[seq_stat_cols].values.astype(np.float32)
        
        # Condition features (one-hot encoded)
        self.variety_data = df[variety_cols].values.astype(np.float32)
        self.tissue_data = df[tissue_cols].values.astype(np.float32)
        self.stage_data = df[stage_cols].values.astype(np.float32)
        
        # Labels
        self.labels = df['label'].values.astype(np.float32)
        
        # Normalize motif features and seq stats
        if fit_scaler:
            self.scaler = StandardScaler()
            combined = np.hstack([self.motif_data, self.seq_stats])
            combined_scaled = self.scaler.fit_transform(combined)
            self.motif_data = combined_scaled[:, :len(motif_cols)]
            self.seq_stats = combined_scaled[:, len(motif_cols):]
        elif scaler is not None:
            self.scaler = scaler
            combined = np.hstack([self.motif_data, self.seq_stats])
            combined_scaled = self.scaler.transform(combined)
            self.motif_data = combined_scaled[:, :len(motif_cols)]
            self.seq_stats = combined_scaled[:, len(motif_cols):]
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'motif': torch.FloatTensor(self.motif_data[idx]),      # (num_motifs,)
            'seq_stats': torch.FloatTensor(self.seq_stats[idx]),   # (4,)
            'variety': torch.FloatTensor(self.variety_data[idx]),  # (num_varieties,)
            'tissue': torch.FloatTensor(self.tissue_data[idx]),    # (num_tissues,)
            'stage': torch.FloatTensor(self.stage_data[idx]),      # (num_stages,)
            'label': torch.FloatTensor([self.labels[idx]])
        }


# =============================================================================
# HYBRID MODELS (Adapted for Global Counts)
# =============================================================================

class HybridLSTM(nn.Module):
    """
    Hybrid LSTM for global motif counts.
    Treats the motif count vector as a sequence.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages, 
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # Motif branch: embed each motif count, process as sequence
        self.motif_embedding = nn.Linear(1, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.motif_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Sequence stats branch
        self.seq_stats_fc = nn.Sequential(
            nn.Linear(num_seq_stats, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 32)
        )
        
        # Variety branch
        self.variety_fc = nn.Sequential(
            nn.Linear(num_varieties, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )
        
        # Tissue branch
        self.tissue_fc = nn.Sequential(
            nn.Linear(num_tissues, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Stage branch
        self.stage_fc = nn.Sequential(
            nn.Linear(num_stages, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Fusion layers
        fusion_dim = hidden_dim + 32 + 64 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Motif branch: (batch, num_motifs) -> (batch, num_motifs, 1) -> embed
        m = motif.unsqueeze(-1)  # (batch, num_motifs, 1)
        m = self.motif_embedding(m)  # (batch, num_motifs, 32)
        lstm_out, (h_n, _) = self.lstm(m)
        motif_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        motif_feat = torch.relu(self.motif_fc(motif_feat))
        
        # Other branches
        seq_feat = self.seq_stats_fc(seq_stats)
        variety_feat = self.variety_fc(variety)
        tissue_feat = self.tissue_fc(tissue)
        stage_feat = self.stage_fc(stage)
        
        # Fusion
        combined = torch.cat([motif_feat, seq_feat, variety_feat, tissue_feat, stage_feat], dim=1)
        out = self.fusion(combined)
        
        return torch.sigmoid(out).squeeze()


class HybridTransformer(nn.Module):
    """
    Transformer for global motif counts.
    Each motif count becomes a token with learned embedding.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.4):
        super().__init__()
        
        # Motif embedding
        self.motif_embedding = nn.Linear(1, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_motifs + 3, hidden_dim) * 0.02)
        
        # Condition embeddings as special tokens
        self.variety_embedding = nn.Linear(num_varieties, hidden_dim)
        self.tissue_embedding = nn.Linear(num_tissues, hidden_dim)
        self.stage_embedding = nn.Linear(num_stages, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Embed motifs: (batch, num_motifs) -> (batch, num_motifs, hidden_dim)
        m = self.motif_embedding(motif.unsqueeze(-1))
        
        # Embed conditions as special tokens
        variety_token = self.variety_embedding(variety).unsqueeze(1)
        tissue_token = self.tissue_embedding(tissue).unsqueeze(1)
        stage_token = self.stage_embedding(stage).unsqueeze(1)
        
        # Concatenate: [VARIETY] [TISSUE] [STAGE] [motif1] ... [motifN]
        x = torch.cat([variety_token, tissue_token, stage_token, m], dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Mean pooling
        pooled = x.mean(dim=1)
        
        # Output
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        
        return torch.sigmoid(out).squeeze()


class HybridCNNAttention(nn.Module):
    """
    1D CNN + self-attention for global motif counts.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # CNN branch for motifs
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Condition branches
        self.variety_fc = nn.Sequential(
            nn.Linear(num_varieties, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.tissue_fc = nn.Sequential(
            nn.Linear(num_tissues, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.stage_fc = nn.Sequential(
            nn.Linear(num_stages, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.seq_stats_fc = nn.Sequential(
            nn.Linear(num_seq_stats, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Fusion
        self.dropout = nn.Dropout(dropout)
        fusion_dim = hidden_dim + 64 + 32 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # CNN on motifs: (batch, num_motifs) -> (batch, 1, num_motifs)
        m = motif.unsqueeze(1)
        m = torch.relu(self.bn1(self.conv1(m)))
        m = torch.relu(self.bn2(self.conv2(m)))
        m = torch.relu(self.bn3(self.conv3(m)))
        
        # Transpose for attention: (batch, num_motifs, hidden_dim)
        m = m.permute(0, 2, 1)
        
        # Self-attention
        attn_out, _ = self.attention(m, m, m)
        m = self.attn_norm(m + attn_out)
        
        # Global pooling
        motif_feat = m.mean(dim=1)
        
        # Condition features
        variety_feat = self.variety_fc(variety)
        tissue_feat = self.tissue_fc(tissue)
        stage_feat = self.stage_fc(stage)
        seq_feat = self.seq_stats_fc(seq_stats)
        
        # Fusion
        combined = torch.cat([motif_feat, variety_feat, tissue_feat, stage_feat, seq_feat], dim=1)
        combined = self.dropout(combined)
        out = self.fusion(combined)
        
        return torch.sigmoid(out).squeeze()


class HybridGatedFusion(nn.Module):
    """
    Gated fusion for global motif counts.
    BEST PERFORMING MODEL with HierarchicalFocal Loss (Test AUC: 0.7850)
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # Motif branch: LSTM
        self.motif_embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=2, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.motif_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Combined condition branch
        cond_input_dim = num_varieties + num_tissues + num_stages + num_seq_stats
        self.condition_fc = nn.Sequential(
            nn.Linear(cond_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Motif features
        m = self.motif_embed(motif.unsqueeze(-1))
        _, (h_n, _) = self.lstm(m)
        motif_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        motif_feat = torch.relu(self.motif_proj(motif_feat))
        
        # Condition features
        cond = torch.cat([variety, tissue, stage, seq_stats], dim=1)
        cond_feat = self.condition_fc(cond)
        
        # Gated fusion
        combined = torch.cat([motif_feat, cond_feat], dim=1)
        gate_weights = self.gate(combined)
        fused = gate_weights * motif_feat + (1 - gate_weights) * cond_feat
        
        # Output
        out = self.output(fused)
        return torch.sigmoid(out).squeeze()


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between motifs and conditions for global counts.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # Motif embedding
        self.motif_embed = nn.Linear(1, hidden_dim)
        self.motif_pos = nn.Parameter(torch.randn(1, num_motifs, hidden_dim) * 0.02)
        
        # Condition embeddings
        self.variety_embed = nn.Linear(num_varieties, hidden_dim)
        self.tissue_embed = nn.Linear(num_tissues, hidden_dim)
        self.stage_embed = nn.Linear(num_stages, hidden_dim)
        
        # Self-attention on motifs
        self.self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention: conditions query motifs
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Sequence stats
        self.seq_stats_fc = nn.Linear(num_seq_stats, hidden_dim // 4)
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Embed motifs with positional encoding
        m = self.motif_embed(motif.unsqueeze(-1)) + self.motif_pos
        
        # Self-attention on motifs
        m_attn, _ = self.self_attn(m, m, m)
        m = self.self_norm(m + m_attn)
        
        # Embed conditions
        variety_emb = self.variety_embed(variety).unsqueeze(1)
        tissue_emb = self.tissue_embed(tissue).unsqueeze(1)
        stage_emb = self.stage_embed(stage).unsqueeze(1)
        cond = torch.cat([variety_emb, tissue_emb, stage_emb], dim=1)
        
        # Cross-attention: conditions attend to motifs
        cond_attn, attn_weights = self.cross_attn(cond, m, m)
        cond = self.cross_norm(cond + cond_attn)
        
        # Pool
        motif_pool = m.mean(dim=1)
        cond_pool = cond.mean(dim=1)
        seq_feat = self.seq_stats_fc(seq_stats)
        
        # Combine and output
        combined = torch.cat([motif_pool, cond_pool, seq_feat], dim=1)
        combined = self.dropout(combined)
        out = self.output(combined)
        
        return torch.sigmoid(out).squeeze()


class HybridMLP(nn.Module):
    """
    Simple MLP baseline for comparison.
    Concatenates all features and processes with dense layers.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        input_dim = num_motifs + num_seq_stats + num_varieties + num_tissues + num_stages
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Concatenate all features
        x = torch.cat([motif, seq_stats, variety, tissue, stage], dim=1)
        out = self.network(x)
        return torch.sigmoid(out).squeeze()


class HybridMultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for global motif counts.
    """
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        # Multi-scale convolutions
        self.conv_k3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(192)
        
        self.conv_merge = nn.Conv1d(192, hidden_dim, kernel_size=3, padding=1)
        self.bn_merge = nn.BatchNorm1d(hidden_dim)
        
        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Condition branches
        self.variety_fc = nn.Linear(num_varieties, 64)
        self.tissue_fc = nn.Linear(num_tissues, 32)
        self.stage_fc = nn.Linear(num_stages, 32)
        self.seq_stats_fc = nn.Linear(num_seq_stats, 32)
        
        # Fusion
        fusion_dim = hidden_dim + 64 + 32 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        # Multi-scale CNN: (batch, num_motifs) -> (batch, 1, num_motifs)
        m = motif.unsqueeze(1)
        
        m3 = torch.relu(self.conv_k3(m))
        m5 = torch.relu(self.conv_k5(m))
        m7 = torch.relu(self.conv_k7(m))
        
        m = torch.cat([m3, m5, m7], dim=1)
        m = self.bn(m)
        
        m = torch.relu(self.bn_merge(self.conv_merge(m)))
        m = m.permute(0, 2, 1)  # (batch, num_motifs, hidden_dim)
        
        # Attention pooling
        attn_weights = self.attn_pool(m)
        motif_feat = (m * attn_weights).sum(dim=1)
        
        # Condition features
        variety_feat = torch.relu(self.variety_fc(variety))
        tissue_feat = torch.relu(self.tissue_fc(tissue))
        stage_feat = torch.relu(self.stage_fc(stage))
        seq_feat = torch.relu(self.seq_stats_fc(seq_stats))
        
        # Fusion
        combined = torch.cat([motif_feat, variety_feat, tissue_feat, stage_feat, seq_feat], dim=1)
        out = self.fusion(combined)
        
        return torch.sigmoid(out).squeeze()


# =============================================================================
# TRAINING WITH HIERARCHICAL FOCAL LOSS
# =============================================================================

def train_model(model, train_loader, dev_loader, criterion, device, 
                epochs, patience, lr, weight_decay):
    """
    Train with HierarchicalFocalLoss, early stopping, and learning rate scheduling.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_auc = 0
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'dev_auc': [], 'dev_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            motif = batch['motif'].to(device)
            seq_stats = batch['seq_stats'].to(device)
            variety = batch['variety'].to(device)
            tissue = batch['tissue'].to(device)
            stage = batch['stage'].to(device)
            label = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(motif, seq_stats, variety, tissue, stage)
            
            # HierarchicalFocalLoss requires condition vectors
            loss = criterion(output, label, variety, tissue, stage)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                motif = batch['motif'].to(device)
                seq_stats = batch['seq_stats'].to(device)
                variety = batch['variety'].to(device)
                tissue = batch['tissue'].to(device)
                stage = batch['stage'].to(device)
                
                output = model(motif, seq_stats, variety, tissue, stage)
                preds.extend(output.cpu().numpy())
                labels.extend(batch['label'].squeeze().numpy())
        
        dev_auc = roc_auc_score(labels, preds)
        dev_acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
        
        history['train_loss'].append(avg_train_loss)
        history['dev_auc'].append(dev_auc)
        history['dev_acc'].append(dev_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | "
                  f"Dev Acc: {dev_acc:.4f} | Dev AUC: {dev_auc:.4f}")
        
        if dev_auc > best_auc:
            best_auc = dev_auc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_auc, history


def evaluate(model, loader, device, name="Dataset"):
    """Comprehensive evaluation with detailed metrics for Q1 journal."""
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score, 
        precision_recall_curve, average_precision_score
    )
    
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            motif = batch['motif'].to(device)
            seq_stats = batch['seq_stats'].to(device)
            variety = batch['variety'].to(device)
            tissue = batch['tissue'].to(device)
            stage = batch['stage'].to(device)
            
            output = model(motif, seq_stats, variety, tissue, stage)
            preds.extend(output.cpu().numpy())
            labels.extend(batch['label'].squeeze().numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    pred_cls = (preds > 0.5).astype(int)
    
    # Core metrics
    acc = accuracy_score(labels, pred_cls)
    bal_acc = balanced_accuracy_score(labels, pred_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred_cls, average='binary')
    auc = roc_auc_score(labels, preds)
    mcc = matthews_corrcoef(labels, pred_cls)
    ap = average_precision_score(labels, preds)
    cm = confusion_matrix(labels, pred_cls)
    
    # Compute specificity and sensitivity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    print(f"\n{'='*60}")
    print(f"{name} Results")
    print('='*60)
    print(f"Accuracy:           {acc:.4f}")
    print(f"Balanced Accuracy:  {bal_acc:.4f}")
    print(f"Precision (PPV):    {prec:.4f}")
    print(f"Recall (Sens):      {rec:.4f}")
    print(f"Specificity:        {specificity:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"PR-AUC (AP):        {ap:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"NPV:                {npv:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Neg      Pos")
    print(f"  Actual Neg  {tn:5d}    {fp:5d}")
    print(f"  Actual Pos  {fn:5d}    {tp:5d}")
    
    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'precision': prec, 'recall': rec, 'specificity': specificity,
        'f1': f1, 'auc': auc, 'pr_auc': ap, 'mcc': mcc,
        'ppv': ppv, 'npv': npv, 'sensitivity': sensitivity,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'predictions': preds, 'labels': labels
    }


def evaluate_per_variety(model, test_df, test_dataset, variety_cols, device, batch_size=128):
    """
    Evaluate model performance separately for each rice variety.
    Returns detailed metrics per variety for Q1 journal publication.
    """
    from sklearn.metrics import (
        matthews_corrcoef, balanced_accuracy_score, average_precision_score
    )
    
    model.eval()
    results_per_variety = {}
    
    # Get variety indices for each sample
    variety_data = test_df[variety_cols].values
    variety_indices = variety_data.argmax(axis=1)
    
    # Get all predictions first
    all_preds = []
    all_labels = []
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    with torch.no_grad():
        for batch in test_loader:
            motif = batch['motif'].to(device)
            seq_stats = batch['seq_stats'].to(device)
            variety = batch['variety'].to(device)
            tissue = batch['tissue'].to(device)
            stage = batch['stage'].to(device)
            
            output = model(motif, seq_stats, variety, tissue, stage)
            all_preds.extend(output.cpu().numpy())
            all_labels.extend(batch['label'].squeeze().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Evaluate per variety
    for v_idx, variety_col in enumerate(variety_cols):
        variety_name = variety_col.replace('Variety_', '')
        
        # Get samples for this variety
        mask = variety_indices == v_idx
        n_samples = mask.sum()
        
        if n_samples < 10:  # Skip varieties with too few samples
            continue
        
        preds = all_preds[mask]
        labels = all_labels[mask]
        pred_cls = (preds > 0.5).astype(int)
        
        # Compute metrics
        acc = accuracy_score(labels, pred_cls)
        
        # Handle edge cases for metrics
        try:
            bal_acc = balanced_accuracy_score(labels, pred_cls)
        except:
            bal_acc = acc
        
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(labels, pred_cls, average='binary', zero_division=0)
        except:
            prec, rec, f1 = 0, 0, 0
        
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = 0.5  # Default when only one class present
        
        try:
            mcc = matthews_corrcoef(labels, pred_cls)
        except:
            mcc = 0
        
        try:
            ap = average_precision_score(labels, preds)
        except:
            ap = 0
        
        cm = confusion_matrix(labels, pred_cls, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Class distribution in this variety
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        
        results_per_variety[variety_name] = {
            'n_samples': int(n_samples),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc,
            'pr_auc': ap,
            'mcc': mcc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return results_per_variety


def print_per_variety_results(variety_results, model_name):
    """
    Print comprehensive per-variety results table for Q1 journal.
    """
    print("\n" + "=" * 130)
    print(f"PER-VARIETY PERFORMANCE ANALYSIS - {model_name}")
    print("=" * 130)
    
    # Sort varieties by sample count (descending)
    sorted_varieties = sorted(variety_results.items(), key=lambda x: x[1]['n_samples'], reverse=True)
    
    # Table header
    print(f"\n{'Variety':<18} {'N':<7} {'Pos/Neg':<10} {'Acc':<8} {'Bal.Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'AUC':<8} {'PR-AUC':<8} {'MCC':<8} {'Sens':<8} {'Spec':<8}")
    print("-" * 130)
    
    for variety_name, metrics in sorted_varieties:
        pos_neg = f"{metrics['n_positive']}/{metrics['n_negative']}"
        print(f"{variety_name:<18} {metrics['n_samples']:<7} {pos_neg:<10} "
              f"{metrics['accuracy']:<8.4f} {metrics['balanced_accuracy']:<8.4f} "
              f"{metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} "
              f"{metrics['auc']:<8.4f} {metrics['pr_auc']:<8.4f} {metrics['mcc']:<8.4f} "
              f"{metrics['sensitivity']:<8.4f} {metrics['specificity']:<8.4f}")
    
    # Statistical summary across varieties
    print("\n" + "-" * 130)
    
    # Weighted average by sample count
    total_samples = sum(v['n_samples'] for v in variety_results.values())
    metrics_to_avg = ['accuracy', 'balanced_accuracy', 'f1', 'auc', 'pr_auc', 'mcc']
    
    weighted_avgs = {}
    simple_avgs = {}
    stds = {}
    
    for metric in metrics_to_avg:
        values = [v[metric] for v in variety_results.values()]
        weights = [v['n_samples'] for v in variety_results.values()]
        
        weighted_avgs[metric] = sum(v * w for v, w in zip(values, weights)) / total_samples
        simple_avgs[metric] = np.mean(values)
        stds[metric] = np.std(values)
    
    print(f"{'WEIGHTED AVG':<18} {total_samples:<7} {'':<10} "
          f"{weighted_avgs['accuracy']:<8.4f} {weighted_avgs['balanced_accuracy']:<8.4f} "
          f"{'-':<8} {'-':<8} {weighted_avgs['f1']:<8.4f} "
          f"{weighted_avgs['auc']:<8.4f} {weighted_avgs['pr_auc']:<8.4f} {weighted_avgs['mcc']:<8.4f}")
    
    print(f"{'MACRO AVG':<18} {'':<7} {'':<10} "
          f"{simple_avgs['accuracy']:<8.4f} {simple_avgs['balanced_accuracy']:<8.4f} "
          f"{'-':<8} {'-':<8} {simple_avgs['f1']:<8.4f} "
          f"{simple_avgs['auc']:<8.4f} {simple_avgs['pr_auc']:<8.4f} {simple_avgs['mcc']:<8.4f}")
    
    print(f"{'STD DEV':<18} {'':<7} {'':<10} "
          f"{stds['accuracy']:<8.4f} {stds['balanced_accuracy']:<8.4f} "
          f"{'-':<8} {'-':<8} {stds['f1']:<8.4f} "
          f"{stds['auc']:<8.4f} {stds['pr_auc']:<8.4f} {stds['mcc']:<8.4f}")
    
    print("=" * 130)
    
    # Find best and worst performing varieties
    best_variety = max(sorted_varieties, key=lambda x: x[1]['auc'])
    worst_variety = min(sorted_varieties, key=lambda x: x[1]['auc'])
    
    print(f"\nBest Performing Variety:  {best_variety[0]} (AUC={best_variety[1]['auc']:.4f}, N={best_variety[1]['n_samples']})")
    print(f"Worst Performing Variety: {worst_variety[0]} (AUC={worst_variety[1]['auc']:.4f}, N={worst_variety[1]['n_samples']})")
    
    return weighted_avgs, simple_avgs, stds


def plot_results(results, output_dir, train_df, variety_cols, tissue_cols, stage_cols):
    """Generate comprehensive Q1 journal-quality visualization plots at 400 DPI."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'lines.linewidth': 2,
        'font.family': 'sans-serif'
    })
    
    DPI = 400
    models = list(results.keys())
    n_models = len(models)
    
    # Color palette for consistency
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    model_colors = {name: colors[i] for i, name in enumerate(models)}
    
    # =========================================================================
    # FIGURE 1: Comprehensive Model Performance Comparison (Multi-metric)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    metrics = ['auc', 'f1', 'mcc', 'balanced_accuracy', 'precision', 'recall']
    metric_labels = ['ROC-AUC', 'F1 Score', 'MCC', 'Balanced Accuracy', 'Precision', 'Recall']
    
    sorted_models = sorted(models, key=lambda m: results[m]['test']['auc'], reverse=True)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        
        dev_vals = [results[m]['dev'][metric] for m in sorted_models]
        test_vals = [results[m]['test'][metric] for m in sorted_models]
        
        x = np.arange(len(sorted_models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, dev_vals, width, label='Validation', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='darkorange', alpha=0.8)
        
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('Hybrid', '') for m in sorted_models], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Model Performance Comparison with HierarchicalFocalLoss', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig1_performance_comparison.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig1_performance_comparison.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 2: ROC Curves for All Models
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 9))
    
    for name in sorted_models:
        res = results[name]
        preds = res['test']['predictions']
        labels = res['test']['labels']
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = res['test']['auc']
        ax.plot(fpr, tpr, color=model_colors[name], lw=2.5, 
                label=f'{name.replace("Hybrid", "")} (AUC={auc_val:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC=0.5000)', alpha=0.7)
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_roc_curves.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig2_roc_curves.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Precision-Recall Curves
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 9))
    
    for name in sorted_models:
        res = results[name]
        preds = res['test']['predictions']
        labels = res['test']['labels']
        precision, recall, _ = precision_recall_curve(labels, preds)
        ap = res['test']['pr_auc']
        ax.plot(recall, precision, color=model_colors[name], lw=2.5, 
                label=f'{name.replace("Hybrid", "")} (AP={ap:.4f})')
    
    # Baseline (no-skill classifier)
    no_skill = sum(results[sorted_models[0]]['test']['labels']) / len(results[sorted_models[0]]['test']['labels'])
    ax.axhline(y=no_skill, color='k', linestyle='--', lw=1.5, label=f'No Skill ({no_skill:.3f})', alpha=0.7)
    
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (PPV)')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_precision_recall_curves.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig3_precision_recall_curves.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 4: Confusion Matrices Heatmap
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    for i, name in enumerate(sorted_models):
        ax = axes[i]
        cm = np.array(results[name]['test']['confusion_matrix'])
        
        # Normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                    cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name.replace("Hybrid", "")}\nAcc={results[name]["test"]["accuracy"]:.3f}')
    
    # Hide unused subplot
    for i in range(len(sorted_models), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig4_confusion_matrices.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig4_confusion_matrices.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Training History (Loss and AUC curves)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(sorted_models):
        ax = axes[i]
        history = results[name]['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax2 = ax.twinx()
        
        line1, = ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.8, lw=2)
        ax.set_ylabel('Loss', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        line2, = ax2.plot(epochs, history['dev_auc'], 'r-', label='Val AUC', alpha=0.8, lw=2)
        ax2.set_ylabel('Validation AUC', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Epoch')
        ax.set_title(f'{name.replace("Hybrid", "")}')
        ax.grid(True, alpha=0.3)
        
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=9)
    
    for i in range(len(sorted_models), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig5_training_history.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig5_training_history.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 6: Calibration Curves (Reliability Diagram)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 9))
    
    for name in sorted_models[:5]:  # Top 5 models
        res = results[name]
        preds = res['test']['predictions']
        labels = res['test']['labels']
        
        prob_true, prob_pred = calibration_curve(labels, preds, n_bins=10, strategy='uniform')
        ax.plot(prob_pred, prob_true, marker='o', markersize=6, color=model_colors[name], 
                lw=2, label=f'{name.replace("Hybrid", "")}')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves (Reliability Diagram)', fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_calibration_curves.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig6_calibration_curves.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 7: Radar/Spider Chart for Best Model vs Others
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    categories = ['ROC-AUC', 'F1', 'MCC', 'Bal. Acc.', 'Precision', 'Recall', 'PR-AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    for name in sorted_models[:4]:  # Top 4 models
        res = results[name]['test']
        values = [res['auc'], res['f1'], res['mcc'], res['balanced_accuracy'], 
                  res['precision'], res['recall'], res['pr_auc']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name.replace('Hybrid', ''), color=model_colors[name])
        ax.fill(angles, values, alpha=0.1, color=model_colors[name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Multi-Metric Performance Comparison (Top 4 Models)', fontweight='bold', size=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_radar_chart.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig7_radar_chart.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 8: Class Distribution (Supplementary)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Variety distribution
    variety_counts = train_df[variety_cols].sum().sort_values(ascending=False)
    variety_names = [c.replace('Variety_', '') for c in variety_counts.index]
    axes[0].barh(variety_names, variety_counts.values, color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Count')
    axes[0].set_title('Rice Variety Distribution (Training Set)', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Tissue distribution
    tissue_counts = train_df[tissue_cols].sum().sort_values(ascending=False)
    tissue_names = [c.replace('Tissue_', '') for c in tissue_counts.index]
    axes[1].barh(tissue_names, tissue_counts.values, color='darkorange', alpha=0.8)
    axes[1].set_xlabel('Count')
    axes[1].set_title('Tissue Type Distribution (Training Set)', fontweight='bold')
    axes[1].invert_yaxis()
    
    # Stage distribution
    stage_counts = train_df[stage_cols].sum().sort_values(ascending=False)
    stage_names = [c.replace('Stage_', '') for c in stage_counts.index]
    axes[2].barh(stage_names, stage_counts.values, color='forestgreen', alpha=0.8)
    axes[2].set_xlabel('Count')
    axes[2].set_title('Developmental Stage Distribution (Training Set)', fontweight='bold')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_class_distribution.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig8_class_distribution.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 9: Prediction Distribution Histograms
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    for i, name in enumerate(sorted_models):
        ax = axes[i]
        res = results[name]['test']
        preds = res['predictions']
        labels = res['labels']
        
        ax.hist(preds[labels == 0], bins=30, alpha=0.6, label='Down-regulated', color='blue', density=True)
        ax.hist(preds[labels == 1], bins=30, alpha=0.6, label='Up-regulated', color='red', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, label='Threshold')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_title(f'{name.replace("Hybrid", "")}')
        ax.legend(loc='upper center', fontsize=8)
    
    for i in range(len(sorted_models), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Prediction Score Distributions (Test Set)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig9_prediction_distributions.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig9_prediction_distributions.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIGURE 10: Performance Metrics Heatmap
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_for_heatmap = ['auc', 'pr_auc', 'f1', 'mcc', 'balanced_accuracy', 'precision', 'recall', 'specificity']
    metric_labels_hm = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC', 'Bal. Acc.', 'Precision', 'Recall', 'Specificity']
    
    data = []
    for name in sorted_models:
        row = [results[name]['test'][m] for m in metrics_for_heatmap]
        data.append(row)
    
    data = np.array(data)
    
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax,
                xticklabels=metric_labels_hm, yticklabels=[m.replace('Hybrid', '') for m in sorted_models],
                vmin=0.5, vmax=0.85, annot_kws={'size': 11})
    ax.set_title('Performance Metrics Heatmap (Test Set)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig10_metrics_heatmap.png'), dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig10_metrics_heatmap.pdf'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n" + "="*70)
    print(f"SAVED Q1 JOURNAL-QUALITY FIGURES ({DPI} DPI) TO: {output_dir}")
    print("="*70)
    print("  - fig1_performance_comparison.png/pdf")
    print("  - fig2_roc_curves.png/pdf")
    print("  - fig3_precision_recall_curves.png/pdf")
    print("  - fig4_confusion_matrices.png/pdf")
    print("  - fig5_training_history.png/pdf")
    print("  - fig6_calibration_curves.png/pdf")
    print("  - fig7_radar_chart.png/pdf")
    print("  - fig8_class_distribution.png/pdf")
    print("  - fig9_prediction_distributions.png/pdf")
    print("  - fig10_metrics_heatmap.png/pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HYBRID ARCHITECTURE FOR MULTI-VARIETY RICE CLASSIFICATION")
    print("Version 2: Optimized with HierarchicalFocalLoss")
    print("(Global Motif Counts - No Positional Binning)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, "dev.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Dev:   {len(dev_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Identify feature columns
    all_cols = train_df.columns.tolist()
    
    # Motif columns (global counts, no __bin suffix)
    motif_cols = [c for c in MOTIF_NAMES if c in all_cols]
    
    # Sequence statistics
    seq_stat_cols = [c for c in SEQ_STATS if c in all_cols]
    
    # Condition columns
    variety_cols = sorted([c for c in all_cols if c.startswith('Variety_')])
    tissue_cols = sorted([c for c in all_cols if c.startswith('Tissue_')])
    stage_cols = sorted([c for c in all_cols if c.startswith('Stage_')])
    
    print(f"\n  Motif features (global): {len(motif_cols)}")
    print(f"  Sequence stat features: {len(seq_stat_cols)}")
    print(f"  Variety features: {len(variety_cols)}")
    print(f"  Tissue features: {len(tissue_cols)}")
    print(f"  Stage features: {len(stage_cols)}")
    print(f"  Total features: {len(motif_cols) + len(seq_stat_cols) + len(variety_cols) + len(tissue_cols) + len(stage_cols)}")
    
    # Print class distribution
    print_class_distribution(train_df, variety_cols, tissue_cols, stage_cols)
    
    # Compute condition weights for HierarchicalFocalLoss
    variety_weights = compute_condition_weights(train_df, variety_cols)
    tissue_weights = compute_condition_weights(train_df, tissue_cols)
    stage_weights = compute_condition_weights(train_df, stage_cols)
    
    # Compute optimal alpha from label distribution
    alpha = compute_alpha_from_labels(train_df)
    
    print(f"\nHierarchicalFocalLoss Configuration:")
    print(f"  Alpha (auto-calculated): {alpha:.3f}")
    print(f"  Gamma: {FOCAL_GAMMA}")
    print(f"  Condition strength: {CONDITION_STRENGTH}")
    
    # Create loss function
    criterion = HierarchicalFocalLoss(
        variety_weights=variety_weights,
        tissue_weights=tissue_weights,
        stage_weights=stage_weights,
        alpha=alpha,
        gamma=FOCAL_GAMMA,
        condition_strength=CONDITION_STRENGTH
    ).to(device)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = HybridDatasetGlobal(
        train_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols,
        fit_scaler=True
    )
    dev_dataset = HybridDatasetGlobal(
        dev_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols,
        scaler=train_dataset.scaler
    )
    test_dataset = HybridDatasetGlobal(
        test_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols,
        scaler=train_dataset.scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Model dimensions
    num_motifs = len(motif_cols)
    num_varieties = len(variety_cols)
    num_tissues = len(tissue_cols)
    num_stages = len(stage_cols)
    num_seq_stats = len(seq_stat_cols)
    
    print(f"\n  Model input dimensions:")
    print(f"    Motifs (global): {num_motifs}")
    print(f"    Varieties: {num_varieties}")
    print(f"    Tissues: {num_tissues}")
    print(f"    Stages: {num_stages}")
    print(f"    Seq stats: {num_seq_stats}")
    
    # Define models
    models = {
        'HybridMLP': HybridMLP(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
        'HybridLSTM': HybridLSTM(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
        'HybridTransformer': HybridTransformer(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, 4, 3, DROPOUT
        ),
        'HybridCNNAttention': HybridCNNAttention(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
        'HybridGatedFusion': HybridGatedFusion(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
        'CrossAttentionFusion': CrossAttentionFusion(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
        'HybridMultiScaleCNN': HybridMultiScaleCNN(
            num_motifs, num_varieties, num_tissues, num_stages,
            num_seq_stats, HIDDEN_DIM, DROPOUT
        ),
    }
    
    # Train all models with HierarchicalFocalLoss
    results = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*70}")
        print(f"TRAINING: {name} + HierarchicalFocalLoss")
        print(f"  Parameters: {n_params:,}")
        print('='*70)
        
        trained_model, best_auc, history = train_model(
            model, train_loader, dev_loader, criterion, device,
            NUM_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY
        )
        
        dev_res = evaluate(trained_model, dev_loader, device, f"{name} - Dev")
        test_res = evaluate(trained_model, test_loader, device, f"{name} - Test")
        
        results[name] = {
            'dev': dev_res,
            'test': test_res,
            'history': history,
            'n_params': n_params
        }
    
    # =========================================================================
    # COMPREHENSIVE RESULTS SUMMARY FOR Q1 JOURNAL
    # =========================================================================
    
    # Sort by test AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test']['auc'], reverse=True)
    
    print("\n")
    print("#" * 90)
    print("#" + " COMPREHENSIVE RESULTS SUMMARY - Q1 JOURNAL PUBLICATION ".center(88) + "#")
    print("#" * 90)
    
    # Table 1: Model Architecture Summary
    print("\n" + "=" * 90)
    print("TABLE 1: MODEL ARCHITECTURE SUMMARY")
    print("=" * 90)
    print(f"{'Rank':<6} {'Model':<25} {'Parameters':<15} {'Loss Function':<25}")
    print("-" * 90)
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {name:<25} {res['n_params']:>12,}   {'HierarchicalFocalLoss':<25}")
    
    # Table 2: Classification Performance (Validation Set)
    print("\n" + "=" * 90)
    print("TABLE 2: CLASSIFICATION PERFORMANCE - VALIDATION SET")
    print("=" * 90)
    print(f"{'Rank':<6} {'Model':<22} {'Accuracy':<10} {'Bal.Acc':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<8}")
    print("-" * 110)
    for rank, (name, res) in enumerate(sorted_results, 1):
        d = res['dev']
        print(f"{rank:<6} {name:<22} {d['accuracy']:<10.4f} {d['balanced_accuracy']:<10.4f} "
              f"{d['precision']:<8.4f} {d['recall']:<8.4f} {d['f1']:<8.4f} {d['auc']:<10.4f} {d['pr_auc']:<10.4f} {d['mcc']:<8.4f}")
    
    # Table 3: Classification Performance (Test Set)
    print("\n" + "=" * 90)
    print("TABLE 3: CLASSIFICATION PERFORMANCE - TEST SET")
    print("=" * 90)
    print(f"{'Rank':<6} {'Model':<22} {'Accuracy':<10} {'Bal.Acc':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'ROC-AUC':<10} {'PR-AUC':<10} {'MCC':<8}")
    print("-" * 110)
    for rank, (name, res) in enumerate(sorted_results, 1):
        t = res['test']
        print(f"{rank:<6} {name:<22} {t['accuracy']:<10.4f} {t['balanced_accuracy']:<10.4f} "
              f"{t['precision']:<8.4f} {t['recall']:<8.4f} {t['f1']:<8.4f} {t['auc']:<10.4f} {t['pr_auc']:<10.4f} {t['mcc']:<8.4f}")
    
    # Table 4: Detailed Confusion Matrix Metrics
    print("\n" + "=" * 90)
    print("TABLE 4: CONFUSION MATRIX ANALYSIS - TEST SET")
    print("=" * 90)
    print(f"{'Rank':<6} {'Model':<22} {'TP':<8} {'TN':<8} {'FP':<8} {'FN':<8} {'Sens':<10} {'Spec':<10} {'PPV':<10} {'NPV':<10}")
    print("-" * 110)
    for rank, (name, res) in enumerate(sorted_results, 1):
        t = res['test']
        print(f"{rank:<6} {name:<22} {t['tp']:<8} {t['tn']:<8} {t['fp']:<8} {t['fn']:<8} "
              f"{t['sensitivity']:<10.4f} {t['specificity']:<10.4f} {t['ppv']:<10.4f} {t['npv']:<10.4f}")
    
    # Best Model Summary
    best_name, best_res = sorted_results[0]
    print("\n" + "=" * 90)
    print("BEST PERFORMING MODEL SUMMARY")
    print("=" * 90)
    print(f"Model:              {best_name}")
    print(f"Loss Function:      HierarchicalFocalLoss (gamma={FOCAL_GAMMA}, alpha={alpha:.3f}, strength={CONDITION_STRENGTH})")
    print(f"Parameters:         {best_res['n_params']:,}")
    print(f"")
    print(f"Validation Metrics:")
    print(f"  - ROC-AUC:        {best_res['dev']['auc']:.4f}")
    print(f"  - PR-AUC:         {best_res['dev']['pr_auc']:.4f}")
    print(f"  - F1 Score:       {best_res['dev']['f1']:.4f}")
    print(f"  - MCC:            {best_res['dev']['mcc']:.4f}")
    print(f"")
    print(f"Test Metrics:")
    print(f"  - ROC-AUC:        {best_res['test']['auc']:.4f}")
    print(f"  - PR-AUC:         {best_res['test']['pr_auc']:.4f}")
    print(f"  - F1 Score:       {best_res['test']['f1']:.4f}")
    print(f"  - MCC:            {best_res['test']['mcc']:.4f}")
    print(f"  - Accuracy:       {best_res['test']['accuracy']:.4f}")
    print(f"  - Sensitivity:    {best_res['test']['sensitivity']:.4f}")
    print(f"  - Specificity:    {best_res['test']['specificity']:.4f}")
    
    # Statistical summary
    print("\n" + "=" * 90)
    print("STATISTICAL SUMMARY ACROSS ALL MODELS - TEST SET")
    print("=" * 90)
    
    metrics_summary = ['auc', 'pr_auc', 'f1', 'mcc', 'accuracy', 'balanced_accuracy']
    metric_names = ['ROC-AUC', 'PR-AUC', 'F1 Score', 'MCC', 'Accuracy', 'Balanced Accuracy']
    
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Range':<10}")
    print("-" * 70)
    for metric, name_m in zip(metrics_summary, metric_names):
        vals = [res['test'][metric] for _, res in sorted_results]
        mean_v, std_v = np.mean(vals), np.std(vals)
        min_v, max_v = np.min(vals), np.max(vals)
        print(f"{name_m:<20} {mean_v:<10.4f} {std_v:<10.4f} {min_v:<10.4f} {max_v:<10.4f} {max_v-min_v:<10.4f}")
    
    # Generate visualization plots
    plot_results(results, OUTPUT_DIR, train_df, variety_cols, tissue_cols, stage_cols)
    
    # =========================================================================
    # PER-VARIETY EVALUATION FOR BEST MODEL
    # =========================================================================
    
    print("\n")
    print("#" * 130)
    print("#" + " PER-VARIETY PERFORMANCE EVALUATION ".center(128) + "#")
    print("#" * 130)
    
    # Evaluate top 3 models per variety using stored predictions
    all_variety_results = {}
    
    for rank, (model_name, res) in enumerate(sorted_results[:3], 1):  # Top 3 models
        # Use predictions already stored in results (no need to reload model)
        preds = res['test']['predictions']
        labels = res['test']['labels']
        
        # Get variety indices for test set
        variety_data = test_df[variety_cols].values
        variety_indices = variety_data.argmax(axis=1)
        
        variety_results = {}
        for v_idx, variety_col in enumerate(variety_cols):
            variety_name = variety_col.replace('Variety_', '')
            
            mask = variety_indices == v_idx
            n_samples = mask.sum()
            
            if n_samples < 5:
                continue
            
            v_preds = preds[mask]
            v_labels = labels[mask]
            pred_cls = (v_preds > 0.5).astype(int)
            
            from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, average_precision_score
            
            acc = accuracy_score(v_labels, pred_cls)
            
            try:
                bal_acc = balanced_accuracy_score(v_labels, pred_cls)
            except:
                bal_acc = acc
            
            try:
                prec, rec, f1, _ = precision_recall_fscore_support(v_labels, pred_cls, average='binary', zero_division=0)
            except:
                prec, rec, f1 = 0, 0, 0
            
            try:
                auc = roc_auc_score(v_labels, v_preds)
            except:
                auc = 0.5
            
            try:
                mcc = matthews_corrcoef(v_labels, pred_cls)
            except:
                mcc = 0
            
            try:
                ap = average_precision_score(v_labels, v_preds)
            except:
                ap = 0
            
            cm = confusion_matrix(v_labels, pred_cls, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            n_pos = int(v_labels.sum())
            n_neg = int(len(v_labels) - n_pos)
            
            variety_results[variety_name] = {
                'n_samples': int(n_samples),
                'n_positive': n_pos,
                'n_negative': n_neg,
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc,
                'pr_auc': ap,
                'mcc': mcc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        all_variety_results[model_name] = variety_results
        print_per_variety_results(variety_results, f"{model_name} (Rank #{rank})")
    
    # Summary table comparing all models across varieties
    print("\n" + "=" * 130)
    print("CROSS-VARIETY PERFORMANCE COMPARISON (TOP 3 MODELS)")
    print("=" * 130)
    
    # Get common varieties
    all_varieties = set()
    for model_name, var_res in all_variety_results.items():
        all_varieties.update(var_res.keys())
    all_varieties = sorted(all_varieties)
    
    print(f"\n{'Variety':<18}", end='')
    for model_name in list(all_variety_results.keys()):
        print(f" {model_name[:15]:<16}", end='')
    print()
    print("-" * (18 + 16 * len(all_variety_results)))
    
    for variety in all_varieties:
        print(f"{variety:<18}", end='')
        for model_name, var_res in all_variety_results.items():
            if variety in var_res:
                auc_val = var_res[variety]['auc']
                print(f" {auc_val:<16.4f}", end='')
            else:
                print(f" {'N/A':<16}", end='')
        print()
    
    print("=" * 130)
    
    # Save results summary JSON only (no model saving)
    summary = {
        'timestamp': datetime.now().isoformat(),
        'approach': 'Global Motif Counts with HierarchicalFocalLoss',
        'loss_type': 'HierarchicalFocalLoss',
        'loss_config': {
            'alpha': alpha,
            'gamma': FOCAL_GAMMA,
            'condition_strength': CONDITION_STRENGTH
        },
        'training_config': {
            'hidden_dim': HIDDEN_DIM,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_epochs': NUM_EPOCHS,
            'patience': PATIENCE
        },
        'data_stats': {
            'train_samples': len(train_df),
            'dev_samples': len(dev_df),
            'test_samples': len(test_df),
            'num_motifs': num_motifs,
            'num_varieties': num_varieties,
            'num_tissues': num_tissues,
            'num_stages': num_stages
        },
        'best_model': sorted_results[0][0],
        'results': {name: {
            'n_params': res['n_params'],
            'dev': {k: v for k, v in res['dev'].items() if k not in ['predictions', 'labels']}, 
            'test': {k: v for k, v in res['test'].items() if k not in ['predictions', 'labels']}
        } for name, res in results.items()},
        'per_variety_results': all_variety_results
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n" + "="*90)
    print(f"OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("="*90)
    print("  - results_summary.json")
    print("  - fig1_performance_comparison.png/pdf")
    print("  - fig2_roc_curves.png/pdf")
    print("  - fig3_precision_recall_curves.png/pdf")
    print("  - fig4_confusion_matrices.png/pdf")
    print("  - fig5_training_history.png/pdf")
    print("  - fig6_calibration_curves.png/pdf")
    print("  - fig7_radar_chart.png/pdf")
    print("  - fig8_class_distribution.png/pdf")
    print("  - fig9_prediction_distributions.png/pdf")
    print("  - fig10_metrics_heatmap.png/pdf")
    
    print("\n" + "#" * 90)
    print("#" + " EXPERIMENT COMPLETE - READY FOR Q1 JOURNAL SUBMISSION ".center(88) + "#")
    print("#" * 90)


if __name__ == '__main__':
    main()
