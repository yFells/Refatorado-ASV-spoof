#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline mínimo e robusto para ASVspoof com 30% dos dados.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import librosa
import soundfile as sf


# 1. EXTRAÇÃO SIMPLES
def extract_simple_features(audio_path, sr=16000):
    """Extrai apenas MFCC de um arquivo de áudio."""
    try:
        # Tentar carregar o áudio
        try:
            audio, _ = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
        except:
            audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        
        # Extrair MFCC simples
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        return mfcc
    except Exception as e:
        print(f"Erro em {audio_path}: {e}")
        return None


def prepare_data(dataset='PA', sample_ratio=0.3):
    """Prepara os dados com amostragem."""
    # Caminhos
    if dataset == 'PA':
        audio_dir = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_train/flac"
        protocol_file = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt"
    else:
        audio_dir = "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_train/flac"
        protocol_file = "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
    
    # Ler protocolo
    genuine_files = []
    spoof_files = []
    
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                file_id = parts[1]  # Segunda coluna
                label = parts[-1]   # Última coluna
                
                if label == 'bonafide':
                    genuine_files.append(file_id)
                else:
                    spoof_files.append(file_id)
    
    print(f"Total - Genuine: {len(genuine_files)}, Spoof: {len(spoof_files)}")
    
    # Amostrar
    n_genuine = int(len(genuine_files) * sample_ratio)
    n_spoof = int(len(spoof_files) * sample_ratio)
    
    sampled_genuine = random.sample(genuine_files, n_genuine)
    sampled_spoof = random.sample(spoof_files, n_spoof)
    
    print(f"Amostra - Genuine: {n_genuine}, Spoof: {n_spoof}")
    
    # Preparar dados
    X = []
    y = []
    
    # Processar genuine
    for file_id in tqdm(sampled_genuine, desc="Processando genuine"):
        audio_path = os.path.join(audio_dir, f"{file_id}.flac")
        if os.path.exists(audio_path):
            features = extract_simple_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 = genuine
    
    # Processar spoof
    for file_id in tqdm(sampled_spoof, desc="Processando spoof"):
        audio_path = os.path.join(audio_dir, f"{file_id}.flac")
        if os.path.exists(audio_path):
            features = extract_simple_features(audio_path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 = spoof
    
    print(f"Total processado: {len(X)} amostras")
    
    return X, y


# 2. MODELO SIMPLES
class SimpleCNN(nn.Module):
    def __init__(self, input_dim=40):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


# 3. DATASET SIMPLES
class SimpleDataset(Dataset):
    def __init__(self, X, y, max_len=400):
        self.X = X
        self.y = y
        self.max_len = max_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        
        # Ajustar tamanho
        if features.shape[1] > self.max_len:
            features = features[:, :self.max_len]
        elif features.shape[1] < self.max_len:
            pad = self.max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad)), 'constant')
        
        return torch.FloatTensor(features), label


# 4. TREINAMENTO SIMPLES
def train_model(X_train, y_train, epochs=10):
    """Treina o modelo."""
    # Dividir treino/validação
    n_train = int(len(X_train) * 0.8)
    indices = list(range(len(X_train)))
    random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_t = [X_train[i] for i in train_indices]
    y_t = [y_train[i] for i in train_indices]
    X_v = [X_train[i] for i in val_indices]
    y_v = [y_train[i] for i in val_indices]
    
    # Datasets e loaders
    train_dataset = SimpleDataset(X_t, y_t)
    val_dataset = SimpleDataset(X_v, y_v)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTreinando no dispositivo: {device}")
    
    # Treinar
    for epoch in range(epochs):
        # Treino
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validação
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        print(f"Época {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
    
    return model


# 5. PIPELINE PRINCIPAL
def main():
    print("=== PIPELINE MÍNIMO ASVspoof ===")
    
    # Preparar dados
    print("\n1. Preparando dados...")
    X, y = prepare_data(dataset='PA', sample_ratio=0.3)
    
    if len(X) == 0:
        print("ERRO: Nenhum dado foi processado!")
        return
    
    # Treinar modelo
    print("\n2. Treinando modelo...")
    model = train_model(X, y, epochs=10)
    
    # Salvar modelo
    print("\n3. Salvando modelo...")
    torch.save(model.state_dict(), 'modelo_simples.pth')
    
    print("\nPipeline concluído!")
    print(f"Modelo salvo em: modelo_simples.pth")


if __name__ == "__main__":
    main()
