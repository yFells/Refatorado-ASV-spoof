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
        
        # Verificar se o diretório existe
        if not os.path.exists(features_dir):
            raise ValueError(f"Diretório de características não encontrado: {features_dir}")
        
        # Listar arquivos .npz disponíveis
        available_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
        print(f"Arquivos .npz encontrados em {features_dir}: {len(available_files)}")
        
        # Carregar labels
        labels_count = 0
        matched_count = 0
        
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels_count += 1
                    
                    # Verificar o formato
                    if len(parts) >= 5:
                        # Formato original do ASVspoof
                        file_id = parts[1]  # Segunda coluna
                        label_str = parts[-1].lower()
                    else:
                        # Formato convertido simples
                        file_id = parts[0]  # Primeira coluna
                        label_str = parts[1].lower()
                    
                    # Determinar o label
                    if label_str in ['bonafide', 'genuine']:
                        label = 0  # genuine
                    else:
                        label = 1  # spoof
                    
                    feature_path = os.path.join(features_dir, f"{file_id}.npz")
                    if os.path.exists(feature_path):
                        self.data.append((feature_path, label))
                        matched_count += 1
        
        print(f"Labels no arquivo: {labels_count}")
        print(f"Arquivos correspondentes encontrados: {matched_count}")
        print(f"Dataset final: {len(self.data)} amostras")
        
        if len(self.data) == 0:
            # Tentar mostrar alguns exemplos para debug
            print("\nExemplos de nomes de arquivos .npz:")
            for i, f in enumerate(available_files[:5]):
                print(f"  {f}")
            
            print("\nExemplos de IDs nos labels:")
            with open(labels_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        if len(parts) >= 5:
                            print(f"  {parts[1]}")
                        else:
                            print(f"  {parts[0]}")
            
            raise ValueError("Nenhuma correspondência encontrada entre labels e características!")
    
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