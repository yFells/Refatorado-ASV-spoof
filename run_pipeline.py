#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal para executar todo o pipeline de análise ASVspoof com 30% dos dados.
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime


def create_config(config_path):
    """Cria arquivo de configuração com os caminhos dos datasets."""
    config = {
        "datasets": {
            "ASVspoof2019PA": {
                "train_audio_dir": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_train/flac",
                "dev_audio_dir": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_dev/flac",
                "eval_audio_dir": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_eval/flac",
                "train_labels_file": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt",
                "dev_labels_file": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt",
                "eval_labels_file": "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt"
            },
            "ASVspoof2019LA": {
                "train_audio_dir": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_train/flac",
                "dev_audio_dir": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_dev/flac",
                "eval_audio_dir": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_eval/flac",
                "train_labels_file": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt",
                "dev_labels_file": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
                "eval_labels_file": "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
            }
        },
        "sample_proportion": 0.3,
        "batch_size": 32,
        "num_epochs": 30,
        "learning_rate": 0.001
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuração salva em: {config_path}")
    return config


def run_command(cmd, description):
    """Executa um comando e mostra o progresso."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} concluído com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erro ao executar {description}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Pipeline ASVspoof com 30% dos dados")
    
    parser.add_argument('--dataset', type=str, default='PA', choices=['PA', 'LA'],
                        help='Dataset a ser usado (PA ou LA)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Diretório base para saída')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Pular extração de características')
    parser.add_argument('--skip-training', action='store_true',
                        help='Pular treinamento')
    parser.add_argument('--skip-testing', action='store_true',
                        help='Pular teste')
    parser.add_argument('--convert-labels', action='store_true',
                        help='Converter labels antes de processar')
    
    args = parser.parse_args()
    
    # Criar estrutura de diretórios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"experiment_{args.dataset}_{timestamp}")
    features_dir = os.path.join(experiment_dir, "features")
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    results_dir = os.path.join(experiment_dir, "results")
    labels_dir = os.path.join(experiment_dir, "labels")
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Criar configuração
    config_path = os.path.join(experiment_dir, "config.json")
    config = create_config(config_path)
    
    # Selecionar dataset
    dataset_key = f"ASVspoof2019{args.dataset}"
    dataset_config = config["datasets"][dataset_key]
    
    # Converter labels se solicitado
    if args.convert_labels:
        print(f"\n📝 Convertendo labels...")
        cmd = [
            sys.executable, "convert_protocols_simple.py",
            "--dataset", args.dataset,
            "--output-dir", labels_dir
        ]
        if run_command(cmd, "Conversão de labels"):
            # Atualizar caminhos para usar labels convertidos
            dataset_config["train_labels_file"] = os.path.join(labels_dir, args.dataset, "train_labels.txt")
            dataset_config["dev_labels_file"] = os.path.join(labels_dir, args.dataset, "dev_labels.txt")
            dataset_config["eval_labels_file"] = os.path.join(labels_dir, args.dataset, "eval_labels.txt")
    
    print(f"\n🚀 Iniciando pipeline para ASVspoof 2019 {args.dataset}")
    print(f"📊 Usando 30% dos dados (mantendo proporção genuíno/spoof)")
    print(f"📁 Diretório do experimento: {experiment_dir}")
    
    # 1. Extração de características
    if not args.skip_extraction:
        cmd = [
            sys.executable, "feature_extraction_fix.py",
            "--train-audio-dir", dataset_config["train_audio_dir"],
            "--dev-audio-dir", dataset_config["dev_audio_dir"],
            "--eval-audio-dir", dataset_config["eval_audio_dir"],
            "--output-dir", features_dir,
            "--train-labels-file", dataset_config["train_labels_file"],
            "--dev-labels-file", dataset_config["dev_labels_file"],
            "--eval-labels-file", dataset_config["eval_labels_file"],
            "--sample-proportion", str(config["sample_proportion"])
        ]
        
        if not run_command(cmd, "Extração de características"):
            print("Erro na extração. Abortando...")
            return 1
    
    # 2. Treinamento
    if not args.skip_training:
        cmd = [
            sys.executable, "train_simple.py",
            "--train-features-dir", os.path.join(features_dir, "train"),
            "--dev-features-dir", os.path.join(features_dir, "dev"),
            "--train-labels-file", dataset_config["train_labels_file"],
            "--dev-labels-file", dataset_config["dev_labels_file"],
            "--batch-size", str(config["batch_size"]),
            "--num-epochs", str(config["num_epochs"]),
            "--learning-rate", str(config["learning_rate"]),
            "--save-dir", checkpoints_dir
        ]
        
        if not run_command(cmd, "Treinamento do modelo"):
            print("Erro no treinamento. Abortando...")
            return 1
    
    # 3. Teste
    if not args.skip_testing:
        checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint não encontrado: {checkpoint_path}")
            return 1
        
        cmd = [
            sys.executable, "test_simple.py",
            "--test-features-dir", os.path.join(features_dir, "eval"),
            "--test-labels-file", dataset_config["eval_labels_file"],
            "--checkpoint", checkpoint_path,
            "--batch-size", str(config["batch_size"]),
            "--results-dir", results_dir
        ]
        
        if not run_command(cmd, "Teste do modelo"):
            print("Erro no teste.")
            return 1
    
    # Resumo final
    print(f"\n{'='*60}")
    print("📊 PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"{'='*60}")
    print(f"Dataset: ASVspoof 2019 {args.dataset}")
    print(f"Proporção de dados: 30%")
    print(f"Diretório do experimento: {experiment_dir}")
    print(f"\n📁 Estrutura de saída:")
    print(f"  ├── features/       # Características extraídas")
    print(f"  ├── checkpoints/    # Modelos treinados")
    print(f"  └── results/        # Resultados do teste")
    
    # Ler e mostrar resultados se disponíveis
    results_file = os.path.join(results_dir, "test_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📈 Resultados do teste:")
        print(f"  • EER: {results['EER']:.4f}")
        print(f"  • AUC: {results['AUC']:.4f}")
        print(f"  • Acurácia: {results['Accuracy']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())