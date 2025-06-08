#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar o formato dos arquivos de protocolo/labels.
"""

import os
import sys


def check_protocol_file(protocol_file):
    """Verifica e mostra o formato do arquivo de protocolo."""
    print(f"\n{'='*60}")
    print(f"Analisando: {protocol_file}")
    print(f"{'='*60}")
    
    if not os.path.exists(protocol_file):
        print(f"ERRO: Arquivo não encontrado!")
        return
    
    with open(protocol_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total de linhas: {len(lines)}")
    
    # Mostrar primeiras 10 linhas
    print("\nPrimeiras 10 linhas:")
    print("-" * 60)
    for i, line in enumerate(lines[:10]):
        parts = line.strip().split()
        print(f"Linha {i+1}: {len(parts)} colunas")
        print(f"  Conteúdo: {line.strip()}")
        if len(parts) >= 2:
            print(f"  Coluna 1: {parts[0]}")
            print(f"  Coluna 2: {parts[1]}")
            if len(parts) >= 4:
                print(f"  Última coluna: {parts[-1]}")
        print()
    
    # Estatísticas
    genuine_count = 0
    spoof_count = 0
    
    for line in lines:
        if 'bonafide' in line.lower():
            genuine_count += 1
        elif 'spoof' in line.lower():
            spoof_count += 1
    
    print(f"\nEstatísticas:")
    print(f"  Genuine/Bonafide: {genuine_count}")
    print(f"  Spoof: {spoof_count}")
    print(f"  Total: {genuine_count + spoof_count}")


def check_audio_dir(audio_dir):
    """Verifica os arquivos de áudio disponíveis."""
    print(f"\n{'='*60}")
    print(f"Verificando diretório de áudio: {audio_dir}")
    print(f"{'='*60}")
    
    if not os.path.exists(audio_dir):
        print(f"ERRO: Diretório não encontrado!")
        return
    
    # Contar arquivos FLAC
    flac_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_files.append(file)
    
    print(f"Total de arquivos .flac encontrados: {len(flac_files)}")
    
    if flac_files:
        print("\nPrimeiros 10 arquivos:")
        for i, file in enumerate(flac_files[:10]):
            print(f"  {i+1}. {file}")


def main():
    # Caminhos para verificar
    pa_train_protocol = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt"
    pa_train_audio = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_train/flac"
    
    la_train_protocol = "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trl.txt"
    la_train_audio = "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_train/flac"
    
    print("VERIFICAÇÃO DE FORMATO DOS DADOS ASVspoof 2019")
    
    # Verificar PA
    if os.path.exists(pa_train_protocol):
        check_protocol_file(pa_train_protocol)
        check_audio_dir(pa_train_audio)
    else:
        print(f"\nDataset PA não encontrado em: {pa_train_protocol}")
    
    # Verificar LA
    if os.path.exists(la_train_protocol):
        check_protocol_file(la_train_protocol)
        check_audio_dir(la_train_audio)
    else:
        print(f"\nDataset LA não encontrado em: {la_train_protocol}")


if __name__ == "__main__":
    main()
