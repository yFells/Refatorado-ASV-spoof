#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simplificado para teste do modelo.
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

from simple_model import SimpleReplayDetector, SimpleDataset


def compute_eer(scores, labels):
    """Calcula Equal Error Rate."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = np.mean([fpr[eer_idx], fnr[eer_idx]])
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold, fpr, tpr


def test_model(model, test_loader, device):
    """Testa o modelo."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_results(scores, labels, save_dir):
    """Plota resultados da avaliação."""
    # Calcular métricas
    eer, threshold, fpr, tpr = compute_eer(scores, labels)
    auc = roc_auc_score(labels, scores)
    
    # Criar figura com subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_xlabel('Taxa de Falso Positivo')
    axes[0].set_ylabel('Taxa de Verdadeiro Positivo')
    axes[0].set_title(f'Curva ROC (EER = {eer:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribuição de scores
    genuine_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]
    
    axes[1].hist(genuine_scores, bins=50, alpha=0.5, label='Genuíno', color='green')
    axes[1].hist(spoof_scores, bins=50, alpha=0.5, label='Spoof', color='red')
    axes[1].axvline(threshold, color='black', linestyle='--', label=f'Limiar EER = {threshold:.3f}')
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequência')
    axes[1].set_title('Distribuição de Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results.png'), dpi=300)
    plt.close()
    
    return eer, auc


def main():
    parser = argparse.ArgumentParser(description="Teste do modelo")
    
    parser.add_argument('--test-features-dir', type=str, required=True)
    parser.add_argument('--test-labels-file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--results-dir', type=str, default='results')
    
    args = parser.parse_args()
    
    # Criar diretório de resultados
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Dataset e DataLoader
    test_dataset = SimpleDataset(args.test_features_dir, args.test_labels_file)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dados de teste: {len(test_dataset)} amostras")
    
    # Carregar modelo
    model = SimpleReplayDetector().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Modelo carregado da época {checkpoint['epoch']+1}")
    print(f"EER de validação: {checkpoint['val_eer']:.4f}")
    
    # Testar
    print("\nTestando modelo...")
    scores, labels = test_model(model, test_loader, device)
    
    # Avaliar e plotar
    eer, auc = plot_results(scores, labels, args.results_dir)
    
    # Calcular outras métricas
    predictions = (scores > 0.5).astype(int)
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    # Salvar resultados
    results = {
        'EER': eer,
        'AUC': auc,
        'Accuracy': accuracy,
        'Confusion Matrix': cm.tolist(),
        'Total Genuine': int((labels == 0).sum()),
        'Total Spoof': int((labels == 1).sum())
    }
    
    # Imprimir resultados
    print("\n=== RESULTADOS DO TESTE ===")
    print(f"EER: {eer:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"\nMatriz de Confusão:")
    print(f"  Predito ->  Genuíno  Spoof")
    print(f"Genuíno        {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"Spoof          {cm[1,0]:5d}   {cm[1,1]:5d}")
    
    # Salvar resultados em arquivo
    import json
    with open(os.path.join(args.results_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResultados salvos em: {args.results_dir}")


if __name__ == "__main__":
    main()
