# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hybrid Architecture for Multi-Variety Rice Classification (Top-K Varieties)
============================================================================
Version for Top-K variety pipeline - Optimized with HierarchicalFocalLoss

Usage:
    python rice_hybrid_classifier_topk.py --top_k 3
    python rice_hybrid_classifier_topk.py --top_k 5
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, 
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = "/home/saiful/DeepRice_Task1/motif_based_classification_ALL_DATA/MSc_thesis2/global motif counting based/all_top_k"

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150
PATIENCE = 25
HIDDEN_DIM = 128          
DROPOUT = 0.4
WEIGHT_DECAY = 0.01

# HierarchicalFocalLoss parameters
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
# HIERARCHICAL FOCAL LOSS
# =============================================================================

class HierarchicalFocalLoss(nn.Module):
    """Focal Loss with hierarchical weighting for biological conditions."""
    
    def __init__(self, variety_weights, tissue_weights, stage_weights,
                 alpha=0.25, gamma=2.0, condition_strength=0.3):
        super().__init__()
        
        self.register_buffer('variety_weights', torch.FloatTensor(variety_weights))
        self.register_buffer('tissue_weights', torch.FloatTensor(tissue_weights))
        self.register_buffer('stage_weights', torch.FloatTensor(stage_weights))
        
        self.alpha = alpha
        self.gamma = gamma
        self.condition_strength = condition_strength
        self.name = "HierarchicalFocal"
    
    def forward(self, pred, target, variety_onehot, tissue_onehot, stage_onehot):
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_weight = torch.where(
            target == 1,
            torch.tensor(self.alpha, device=target.device),
            torch.tensor(1 - self.alpha, device=target.device)
        )
        
        v_weight = (variety_onehot * self.variety_weights).sum(dim=1)
        t_weight = (tissue_onehot * self.tissue_weights).sum(dim=1)
        s_weight = (stage_onehot * self.stage_weights).sum(dim=1)
        
        condition_weight = (v_weight * t_weight * s_weight) ** (1/3)
        condition_weight = 1 + self.condition_strength * (condition_weight - 1)
        
        total_weight = alpha_weight * focal_weight * condition_weight
        return (bce * total_weight).mean()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_condition_weights(df, condition_cols):
    """Compute inverse frequency weights for one-hot encoded conditions."""
    counts = df[condition_cols].sum(axis=0).values
    weights = len(df) / (len(condition_cols) * counts)
    weights = weights / weights.mean()
    return weights


def compute_alpha_from_labels(train_df):
    """Compute optimal alpha for focal loss based on label distribution."""
    label_counts = train_df['label'].value_counts()
    n_neg = label_counts.get(0, 0)
    n_pos = label_counts.get(1, 0)
    alpha = n_neg / (n_neg + n_pos)
    return alpha


# =============================================================================
# DATASET
# =============================================================================

class HybridDatasetGlobal(Dataset):
    """Dataset for global motif counts (no binning)."""
    
    def __init__(self, df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols,
                 scaler=None, fit_scaler=False):
        self.motif_data = df[motif_cols].values.astype(np.float32)
        self.seq_stats = df[seq_stat_cols].values.astype(np.float32)
        self.variety_data = df[variety_cols].values.astype(np.float32)
        self.tissue_data = df[tissue_cols].values.astype(np.float32)
        self.stage_data = df[stage_cols].values.astype(np.float32)
        self.labels = df['label'].values.astype(np.float32)
        
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
            'motif': torch.FloatTensor(self.motif_data[idx]),
            'seq_stats': torch.FloatTensor(self.seq_stats[idx]),
            'variety': torch.FloatTensor(self.variety_data[idx]),
            'tissue': torch.FloatTensor(self.tissue_data[idx]),
            'stage': torch.FloatTensor(self.stage_data[idx]),
            'label': torch.FloatTensor([self.labels[idx]])
        }


# =============================================================================
# HYBRID MODELS
# =============================================================================

class HybridLSTM(nn.Module):
    """Hybrid LSTM for global motif counts."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages, 
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.motif_embedding = nn.Linear(1, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers=2,
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.motif_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.seq_stats_fc = nn.Sequential(
            nn.Linear(num_seq_stats, 32), nn.ReLU(), nn.Dropout(dropout / 2), nn.Linear(32, 32))
        self.variety_fc = nn.Sequential(
            nn.Linear(num_varieties, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 64))
        self.tissue_fc = nn.Sequential(
            nn.Linear(num_tissues, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 32))
        self.stage_fc = nn.Sequential(
            nn.Linear(num_stages, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 32))
        
        fusion_dim = hidden_dim + 32 + 64 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = self.motif_embedding(motif.unsqueeze(-1))
        _, (h_n, _) = self.lstm(m)
        motif_feat = torch.relu(self.motif_fc(torch.cat((h_n[-2], h_n[-1]), dim=1)))
        
        combined = torch.cat([motif_feat, self.seq_stats_fc(seq_stats), 
                             self.variety_fc(variety), self.tissue_fc(tissue), self.stage_fc(stage)], dim=1)
        return torch.sigmoid(self.fusion(combined)).squeeze()


class HybridTransformer(nn.Module):
    """Transformer for global motif counts."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.motif_embedding = nn.Linear(1, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_motifs + 3, hidden_dim) * 0.02)
        
        self.variety_embedding = nn.Linear(num_varieties, hidden_dim)
        self.tissue_embedding = nn.Linear(num_tissues, hidden_dim)
        self.stage_embedding = nn.Linear(num_stages, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = self.motif_embedding(motif.unsqueeze(-1))
        variety_token = self.variety_embedding(variety).unsqueeze(1)
        tissue_token = self.tissue_embedding(tissue).unsqueeze(1)
        stage_token = self.stage_embedding(stage).unsqueeze(1)
        
        x = torch.cat([variety_token, tissue_token, stage_token, m], dim=1)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.norm(x)
        pooled = self.dropout(x.mean(dim=1))
        return torch.sigmoid(self.fc(pooled)).squeeze()


class HybridCNNAttention(nn.Module):
    """1D CNN + self-attention for global motif counts."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        self.variety_fc = nn.Sequential(nn.Linear(num_varieties, 64), nn.ReLU(), nn.Linear(64, 64))
        self.tissue_fc = nn.Sequential(nn.Linear(num_tissues, 32), nn.ReLU(), nn.Linear(32, 32))
        self.stage_fc = nn.Sequential(nn.Linear(num_stages, 32), nn.ReLU(), nn.Linear(32, 32))
        self.seq_stats_fc = nn.Sequential(nn.Linear(num_seq_stats, 32), nn.ReLU(), nn.Linear(32, 32))
        
        self.dropout = nn.Dropout(dropout)
        fusion_dim = hidden_dim + 64 + 32 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = motif.unsqueeze(1)
        m = torch.relu(self.bn1(self.conv1(m)))
        m = torch.relu(self.bn2(self.conv2(m)))
        m = torch.relu(self.bn3(self.conv3(m)))
        m = m.permute(0, 2, 1)
        attn_out, _ = self.attention(m, m, m)
        m = self.attn_norm(m + attn_out)
        motif_feat = m.mean(dim=1)
        
        combined = torch.cat([motif_feat, self.variety_fc(variety), self.tissue_fc(tissue),
                             self.stage_fc(stage), self.seq_stats_fc(seq_stats)], dim=1)
        return torch.sigmoid(self.fusion(self.dropout(combined))).squeeze()


class HybridGatedFusion(nn.Module):
    """Gated fusion for global motif counts - BEST PERFORMING MODEL."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.motif_embed = nn.Linear(1, 32)
        self.lstm = nn.LSTM(32, hidden_dim, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True)
        self.motif_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        cond_input_dim = num_varieties + num_tissues + num_stages + num_seq_stats
        self.condition_fc = nn.Sequential(
            nn.Linear(cond_input_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, hidden_dim))
        
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = self.motif_embed(motif.unsqueeze(-1))
        _, (h_n, _) = self.lstm(m)
        motif_feat = torch.relu(self.motif_proj(torch.cat((h_n[-2], h_n[-1]), dim=1)))
        
        cond = torch.cat([variety, tissue, stage, seq_stats], dim=1)
        cond_feat = self.condition_fc(cond)
        
        combined = torch.cat([motif_feat, cond_feat], dim=1)
        gate_weights = self.gate(combined)
        fused = gate_weights * motif_feat + (1 - gate_weights) * cond_feat
        
        return torch.sigmoid(self.output(fused)).squeeze()


class CrossAttentionFusion(nn.Module):
    """Cross-attention between motifs and conditions."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.motif_embed = nn.Linear(1, hidden_dim)
        self.motif_pos = nn.Parameter(torch.randn(1, num_motifs, hidden_dim) * 0.02)
        
        self.variety_embed = nn.Linear(num_varieties, hidden_dim)
        self.tissue_embed = nn.Linear(num_tissues, hidden_dim)
        self.stage_embed = nn.Linear(num_stages, hidden_dim)
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        self.seq_stats_fc = nn.Linear(num_seq_stats, hidden_dim // 4)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = self.motif_embed(motif.unsqueeze(-1)) + self.motif_pos
        m_attn, _ = self.self_attn(m, m, m)
        m = self.self_norm(m + m_attn)
        
        cond = torch.cat([self.variety_embed(variety).unsqueeze(1), 
                         self.tissue_embed(tissue).unsqueeze(1), 
                         self.stage_embed(stage).unsqueeze(1)], dim=1)
        cond_attn, _ = self.cross_attn(cond, m, m)
        cond = self.cross_norm(cond + cond_attn)
        
        combined = torch.cat([m.mean(dim=1), cond.mean(dim=1), self.seq_stats_fc(seq_stats)], dim=1)
        return torch.sigmoid(self.output(self.dropout(combined))).squeeze()


class HybridMLP(nn.Module):
    """Simple MLP baseline."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        input_dim = num_motifs + num_seq_stats + num_varieties + num_tissues + num_stages
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        x = torch.cat([motif, seq_stats, variety, tissue, stage], dim=1)
        return torch.sigmoid(self.network(x)).squeeze()


class HybridMultiScaleCNN(nn.Module):
    """Multi-scale CNN for global motif counts."""
    def __init__(self, num_motifs, num_varieties, num_tissues, num_stages,
                 num_seq_stats=4, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.conv_k3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(192)
        self.conv_merge = nn.Conv1d(192, hidden_dim, kernel_size=3, padding=1)
        self.bn_merge = nn.BatchNorm1d(hidden_dim)
        
        self.attn_pool = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softmax(dim=1))
        
        self.variety_fc = nn.Linear(num_varieties, 64)
        self.tissue_fc = nn.Linear(num_tissues, 32)
        self.stage_fc = nn.Linear(num_stages, 32)
        self.seq_stats_fc = nn.Linear(num_seq_stats, 32)
        
        fusion_dim = hidden_dim + 64 + 32 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1))
    
    def forward(self, motif, seq_stats, variety, tissue, stage):
        m = motif.unsqueeze(1)
        m = torch.cat([torch.relu(self.conv_k3(m)), torch.relu(self.conv_k5(m)), torch.relu(self.conv_k7(m))], dim=1)
        m = torch.relu(self.bn_merge(self.conv_merge(self.bn(m)))).permute(0, 2, 1)
        motif_feat = (m * self.attn_pool(m)).sum(dim=1)
        
        combined = torch.cat([motif_feat, torch.relu(self.variety_fc(variety)), torch.relu(self.tissue_fc(tissue)),
                             torch.relu(self.stage_fc(stage)), torch.relu(self.seq_stats_fc(seq_stats))], dim=1)
        return torch.sigmoid(self.fusion(combined)).squeeze()


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_model(model, train_loader, dev_loader, criterion, device, 
                epochs, patience, lr, weight_decay):
    """Train with HierarchicalFocalLoss and early stopping."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_auc = 0
    patience_counter = 0
    best_state = None
    history = {'train_loss': [], 'dev_auc': [], 'dev_acc': []}
    
    for epoch in range(epochs):
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
            loss = criterion(output, label, variety, tissue, stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                output = model(batch['motif'].to(device), batch['seq_stats'].to(device),
                              batch['variety'].to(device), batch['tissue'].to(device), batch['stage'].to(device))
                # Ensure output is at least 1D for proper extend
                output_np = output.cpu().numpy()
                if output_np.ndim == 0:
                    preds.append(output_np.item())
                else:
                    preds.extend(output_np.flatten().tolist())
                label_np = batch['label'].squeeze().numpy()
                if label_np.ndim == 0:
                    labels.append(label_np.item())
                else:
                    labels.extend(label_np.flatten().tolist())
        
        dev_auc = roc_auc_score(labels, preds)
        dev_acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
        
        history['train_loss'].append(avg_train_loss)
        history['dev_auc'].append(dev_auc)
        history['dev_acc'].append(dev_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val AUC: {dev_auc:.4f}")
        
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
    """Comprehensive evaluation with detailed metrics."""
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            output = model(batch['motif'].to(device), batch['seq_stats'].to(device),
                          batch['variety'].to(device), batch['tissue'].to(device), batch['stage'].to(device))
            # Ensure output is at least 1D for proper extend
            output_np = output.cpu().numpy()
            if output_np.ndim == 0:
                preds.append(output_np.item())
            else:
                preds.extend(output_np.flatten().tolist())
            label_np = batch['label'].squeeze().numpy()
            if label_np.ndim == 0:
                labels.append(label_np.item())
            else:
                labels.extend(label_np.flatten().tolist())
    
    preds = np.array(preds)
    labels = np.array(labels)
    pred_cls = (preds > 0.5).astype(int)
    
    acc = accuracy_score(labels, pred_cls)
    bal_acc = balanced_accuracy_score(labels, pred_cls)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, pred_cls, average='binary')
    auc = roc_auc_score(labels, preds)
    mcc = matthews_corrcoef(labels, pred_cls)
    ap = average_precision_score(labels, preds)
    cm = confusion_matrix(labels, pred_cls)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
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
    
    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc, 'precision': prec, 'recall': rec,
        'specificity': specificity, 'f1': f1, 'auc': auc, 'pr_auc': ap, 'mcc': mcc,
        'ppv': ppv, 'npv': npv, 'sensitivity': sensitivity, 'confusion_matrix': cm.tolist(),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'predictions': preds, 'labels': labels
    }


def plot_results(results, output_dir, top_k):
    """Generate key performance plots at 400 DPI."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10
    })
    DPI = 400
    
    models = list(results.keys())
    sorted_models = sorted(models, key=lambda m: results[m]['test']['auc'], reverse=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    model_colors = {name: colors[i] for i, name in enumerate(models)}
    
    # Figure 1: Performance comparison
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = ['auc', 'f1', 'mcc', 'balanced_accuracy', 'precision', 'recall']
    metric_labels = ['ROC-AUC', 'F1 Score', 'MCC', 'Balanced Accuracy', 'Precision', 'Recall']
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        dev_vals = [results[m]['dev'][metric] for m in sorted_models]
        test_vals = [results[m]['test'][metric] for m in sorted_models]
        x = np.arange(len(sorted_models))
        width = 0.35
        ax.bar(x - width/2, dev_vals, width, label='Validation', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, test_vals, width, label='Test', color='darkorange', alpha=0.8)
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('Hybrid', '') for m in sorted_models], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Model Performance - Top {top_k} Varieties', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, 'fig1_performance_comparison.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 2: ROC Curves
    fig, ax = plt.subplots(figsize=(10, 9))
    for name in sorted_models:
        res = results[name]
        fpr, tpr, _ = roc_curve(res['test']['labels'], res['test']['predictions'])
        ax.plot(fpr, tpr, color=model_colors[name], lw=2.5, 
                label=f'{name.replace("Hybrid", "")} (AUC={res["test"]["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - Top {top_k} Varieties', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_roc_curves.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Confusion matrices for top 3 models
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, name in enumerate(sorted_models[:3]):
        ax = axes[i]
        cm = np.array(results[name]['test']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                    cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name.replace("Hybrid", "")}\nAUC={results[name]["test"]["auc"]:.4f}')
    
    plt.suptitle(f'Confusion Matrices (Top 3 Models) - Top {top_k} Varieties', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'fig3_confusion_matrices.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Figures saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rice Hybrid Classifier for Top-K Varieties')
    parser.add_argument('--top_k', type=int, required=True, help='Number of top varieties (1-13)')
    parser.add_argument('--gpu', type=int, default=4, help='GPU device ID')
    args = parser.parse_args()
    
    top_k = args.top_k
    
    if top_k < 1 or top_k > 13:
        print(f"ERROR: top_k must be between 1 and 13, got {top_k}")
        sys.exit(1)
    
    # Setup paths
    DATA_DIR = os.path.join(BASE_DIR, f"top{top_k}", "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, f"top{top_k}", "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Redirect stdout
    console_file = os.path.join(OUTPUT_DIR, f"console_output_top{top_k}.txt")
    sys.stdout = open(console_file, "w")
    
    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 90)
    print(f"RICE HYBRID CLASSIFIER - TOP {top_k} VARIETIES")
    print("=" * 90)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    print("\n" + "=" * 90)
    print("LOADING DATA")
    print("=" * 90)
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, 'dev.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    print(f"  Train samples: {len(train_df)}")
    print(f"  Dev samples:   {len(dev_df)}")
    print(f"  Test samples:  {len(test_df)}")
    
    # Identify columns
    variety_cols = [c for c in train_df.columns if c.startswith('Variety_')]
    tissue_cols = [c for c in train_df.columns if c.startswith('Tissue_')]
    stage_cols = [c for c in train_df.columns if c.startswith('Stage_')]
    motif_cols = [c for c in MOTIF_NAMES if c in train_df.columns]
    seq_stat_cols = [c for c in SEQ_STATS if c in train_df.columns]
    
    print(f"\n  Varieties: {len(variety_cols)} ({', '.join([c.replace('Variety_', '') for c in variety_cols])})")
    print(f"  Tissues:   {len(tissue_cols)}")
    print(f"  Stages:    {len(stage_cols)}")
    print(f"  Motifs:    {len(motif_cols)}")
    
    # Create datasets
    train_dataset = HybridDatasetGlobal(
        train_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols, fit_scaler=True)
    dev_dataset = HybridDatasetGlobal(
        dev_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols, scaler=train_dataset.scaler)
    test_dataset = HybridDatasetGlobal(
        test_df, motif_cols, seq_stat_cols, variety_cols, tissue_cols, stage_cols, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Compute loss weights
    variety_weights = compute_condition_weights(train_df, variety_cols)
    tissue_weights = compute_condition_weights(train_df, tissue_cols)
    stage_weights = compute_condition_weights(train_df, stage_cols)
    alpha = compute_alpha_from_labels(train_df)
    
    print(f"\n  Focal Loss Alpha: {alpha:.4f}")
    
    # Initialize loss
    criterion = HierarchicalFocalLoss(
        variety_weights, tissue_weights, stage_weights,
        alpha=alpha, gamma=FOCAL_GAMMA, condition_strength=CONDITION_STRENGTH
    ).to(device)
    
    # Model dimensions
    num_motifs = len(motif_cols)
    num_varieties = len(variety_cols)
    num_tissues = len(tissue_cols)
    num_stages = len(stage_cols)
    num_seq_stats = len(seq_stat_cols)
    
    # Define models
    models = {
        'HybridMLP': HybridMLP(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
        'HybridLSTM': HybridLSTM(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
        'HybridTransformer': HybridTransformer(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, 4, 3, DROPOUT),
        'HybridCNNAttention': HybridCNNAttention(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
        'HybridGatedFusion': HybridGatedFusion(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
        'CrossAttentionFusion': CrossAttentionFusion(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
        'HybridMultiScaleCNN': HybridMultiScaleCNN(num_motifs, num_varieties, num_tissues, num_stages, num_seq_stats, HIDDEN_DIM, DROPOUT),
    }
    
    # Train and evaluate all models
    print("\n" + "=" * 90)
    print("TRAINING MODELS")
    print("=" * 90)
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print('='*70)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        
        model, best_val_auc, history = train_model(
            model, train_loader, dev_loader, criterion, device,
            NUM_EPOCHS, PATIENCE, LEARNING_RATE, WEIGHT_DECAY
        )
        
        print(f"\n  Best Validation AUC: {best_val_auc:.4f}")
        
        dev_metrics = evaluate(model, dev_loader, device, f"{name} - Validation")
        test_metrics = evaluate(model, test_loader, device, f"{name} - Test")
        
        results[name] = {
            'n_params': n_params,
            'history': history,
            'dev': dev_metrics,
            'test': test_metrics
        }
    
    # Results summary
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test']['auc'], reverse=True)
    
    print(f"\n{'Model':<25} {'Params':<12} {'Val AUC':<10} {'Test AUC':<10} {'Test F1':<10} {'Test MCC':<10}")
    print("-" * 85)
    for name, res in sorted_results:
        print(f"{name:<25} {res['n_params']:<12,} {res['dev']['auc']:<10.4f} {res['test']['auc']:<10.4f} "
              f"{res['test']['f1']:<10.4f} {res['test']['mcc']:<10.4f}")
    
    best_model_name = sorted_results[0][0]
    best_test_auc = sorted_results[0][1]['test']['auc']
    print(f"\nBest Model: {best_model_name} (Test AUC: {best_test_auc:.4f})")
    
    # Generate plots
    plot_results(results, OUTPUT_DIR, top_k)
    
    # Save results JSON
    summary = {
        'top_k': top_k,
        'timestamp': datetime.now().isoformat(),
        'data_stats': {
            'train_samples': len(train_df),
            'dev_samples': len(dev_df),
            'test_samples': len(test_df),
            'num_varieties': num_varieties,
            'variety_names': [c.replace('Variety_', '') for c in variety_cols],
            'num_motifs': num_motifs,
            'num_tissues': num_tissues,
            'num_stages': num_stages
        },
        'best_model': best_model_name,
        'best_test_auc': best_test_auc,
        'results': {name: {
            'n_params': res['n_params'],
            'dev': {k: v for k, v in res['dev'].items() if k not in ['predictions', 'labels']},
            'test': {k: v for k, v in res['test'].items() if k not in ['predictions', 'labels']}
        } for name, res in results.items()}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n" + "=" * 90)
    print(f"TOP {top_k} EXPERIMENT COMPLETE")
    print("=" * 90)
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Return best AUC for comparison script
    return best_test_auc


if __name__ == '__main__':
    main()
