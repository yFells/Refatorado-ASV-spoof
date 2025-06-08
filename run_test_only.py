#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script auxiliar para executar apenas o teste do modelo treinado.
"""

import os
import sys
import subprocess
import argparse


def find_latest_experiment(base_dir, dataset='PA'):
    """Encontra o experimento mais recente."""
    experiments = []
    
    for item in os.listdir(base_dir):
        if item.startswith(f'experiment_{dataset}_'):
            exp_path = os.path.join(base_dir, item)
            if os.path.isdir(exp_path):
                experiments.append((item, exp_path))
    
    if not experiments:
        return None
    
    # Ordenar por nome (que contém timestamp)
    experiments.sort(reverse=True)
    return experiments[0][1]


def main():
    parser = argparse.ArgumentParser(description="Executar teste do modelo ASVspoof")
    
    parser.add_argument('--experiment-dir', type=str, default=None,
                        help='Diretório do experimento (se não fornecido, usa o mais recente)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Diretório base onde procurar experimentos')
    parser.add_argument('--dataset', type=str, default='PA', choices=['PA', 'LA'],
                        help='Dataset usado no treinamento')
    
    args = parser.parse_args()
    
    # Encontrar diretório do experimento
    if args.experiment_dir:
        experiment_dir = args.experiment_dir
    else:
        experiment_dir = find_latest_experiment(args.output_dir, args.dataset)
        if not experiment_dir:
            print(f"Nenhum experimento encontrado em {args.output_dir}")
            return 1
    
    print(f"Usando experimento: {experiment_dir}")
    
    # Verificar se os arquivos necessários existem
    features_dir = os.path.join(experiment_dir, "features", "eval")
    if args.dataset == 'PA':
        labels_file = os.path.join(experiment_dir, "labels", "PA", "eval_labels.txt")
    else:
        labels_file = os.path.join(experiment_dir, "labels", "LA", "eval_labels.txt")
    checkpoint = os.path.join(experiment_dir, "checkpoints", "best_model.pth")
    results_dir = os.path.join(experiment_dir, "results")
    
    # Verificar existência dos arquivos
    if not os.path.exists(features_dir):
        print(f"Diretório de características não encontrado: {features_dir}")
        return 1
    
    if not os.path.exists(labels_file):
        print(f"Arquivo de labels não encontrado: {labels_file}")
        return 1
    
    if not os.path.exists(checkpoint):
        print(f"Checkpoint não encontrado: {checkpoint}")
        return 1
    
    # Criar diretório de resultados
    os.makedirs(results_dir, exist_ok=True)
    
    # Construir comando
    cmd = [
        sys.executable, "test_simple.py",
        "--test-features-dir", features_dir,
        "--test-labels-file", labels_file,
        "--checkpoint", checkpoint,
        "--batch-size", "32",
        "--results-dir", results_dir
    ]
    
    print("\nExecutando teste...")
    print(f"Comando: {' '.join(cmd)}")
    
    # Executar
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Teste concluído com sucesso!")
        print(f"Resultados salvos em: {results_dir}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erro ao executar teste: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
