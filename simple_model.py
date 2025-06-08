#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo simplificado para detecção de ataques de replay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleReplayDetector(nn.Module):
    """Modelo CNN simplificado para detecção de replay attacks."""
    
    def __init__(self, input_dim=180, hidden_size=256, num_classes=2):
        super(SimpleReplayDetector, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(512, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: [batch, features, time]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleDataset(torch.utils.data.Dataset):
    """Dataset simplificado."""
    
    def __init__(self, features_dir, labels_file, max_length=400):
        import os
        import numpy as np
        
        self.features_dir = features_dir
        self.max_length = max_length
        self.data = []
        
        # Carregar labels
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    label = 1 if parts[1] == 'spoof' else 0
                    
                    feature_path = os.path.join(features_dir, f"{file_id}.npz")
                    if os.path.exists(feature_path):
                        self.data.append((feature_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import numpy as np
        
        feature_path, label = self.data[idx]
        
        # Carregar características
        data = np.load(feature_path)
        mfcc = data['mfcc'] if 'mfcc' in data else np.zeros((90, 100))
        cqcc = data['cqcc'] if 'cqcc' in data else np.zeros((90, 100))
        
        # Concatenar características
        features = np.vstack([mfcc, cqcc])  # Shape: [180, time]
        
        # Ajustar comprimento
        if features.shape[1] > self.max_length:
            features = features[:, :self.max_length]
        elif features.shape[1] < self.max_length:
            pad_width = self.max_length - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), 'constant')
        
        return torch.FloatTensor(features), label
