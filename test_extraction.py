#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para verificar a extração de características.
"""

import os
import sys
from feature_extraction_fix import FeatureExtractor


def test_extraction():
    """Testa a extração com apenas alguns arquivos."""
    
    # Configurar caminhos
    audio_dir = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_train/flac"
    labels_file = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt"
    output_dir = "test_features"
    
    print("=== TESTE DE EXTRAÇÃO ===")
    print(f"Áudio: {audio_dir}")
    print(f"Labels: {labels_file}")
    print(f"Saída: {output_dir}")
    
    # Verificar se os caminhos existem
    if not os.path.exists(audio_dir):
        print(f"ERRO: Diretório de áudio não encontrado: {audio_dir}")
        return
    
    if not os.path.exists(labels_file):
        print(f"ERRO: Arquivo de labels não encontrado: {labels_file}")
        return
    
    # Criar extrator
    extractor = FeatureExtractor()
    
    # Testar extração com proporção muito pequena (0.001 = 0.1%)
    print("\nTestando extração com 0.1% dos dados...")
    extractor.batch_feature_extraction(
        audio_dir, output_dir, labels_file, sample_proportion=0.001
    )
    
    # Verificar o que foi criado
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\nArquivos criados em {output_dir}: {len(files)}")
        if files:
            print("Primeiros 5 arquivos:")
            for f in files[:5]:
                print(f"  - {f}")
    else:
        print(f"\nERRO: Diretório de saída não foi criado!")


if __name__ == "__main__":
    test_extraction()
