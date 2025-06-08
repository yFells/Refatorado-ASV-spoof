#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplificado para treinamento do modelo de detecção de replay attacks.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm

from simple_model import SimpleReplayDetector, SimpleDataset


def compute_eer(scores, labels):
    """Calcula Equal Error Rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = np.mean([fpr[eer_idx], fnr[eer_idx]])
    return eer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Treina por uma época."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels in tqdm(train_loader, desc="Treinando"):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Coletar predições
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_preds.extend(probs.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    eer = compute_eer(all_preds, all_labels)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    
    return avg_loss, eer, acc


def evaluate(model, val_loader, criterion, device):
    """Avalia o modelo."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Validando"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    eer = compute_eer(all_preds, all_labels)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    auc = roc_auc_score(all_labels, all_preds)
    
    return avg_loss, eer, acc, auc


def main():
    parser = argparse.ArgumentParser(description="Treinamento simplificado")
    
    parser.add_argument('--train-features-dir', type=str, required=True)
    parser.add_argument('--dev-features-dir', type=str, required=True)
    parser.add_argument('--train-labels-file', type=str, required=True)
    parser.add_argument('--dev-labels-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    # Criar diretório de checkpoints
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Datasets e DataLoaders
    train_dataset = SimpleDataset(args.train_features_dir, args.train_labels_file)
    dev_dataset = SimpleDataset(args.dev_features_dir, args.dev_labels_file)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dados de treino: {len(train_dataset)} amostras")
    print(f"Dados de validação: {len(dev_dataset)} amostras")
    
    # Modelo
    model = SimpleReplayDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Treinamento
    best_eer = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nÉpoca {epoch+1}/{args.num_epochs}")
        
        # Treinar
        train_loss, train_eer, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validar
        val_loss, val_eer, val_acc, val_auc = evaluate(
            model, dev_loader, criterion, device
        )
        
        # Ajustar learning rate
        scheduler.step(val_eer)
        
        print(f"Treino - Loss: {train_loss:.4f}, EER: {train_eer:.4f}, Acc: {train_acc:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, EER: {val_eer:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
        
        # Salvar melhor modelo
        if val_eer < best_eer:
            best_eer = val_eer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_eer': val_eer,
                'val_auc': val_auc
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Melhor modelo salvo! EER: {val_eer:.4f}")
    
    print(f"\nTreinamento concluído! Melhor EER: {best_eer:.4f}")


if __name__ == "__main__":
    main()
