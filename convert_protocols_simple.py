#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para converter protocolos ASVspoof para formato simples de labels.
"""

import os
import argparse


def convert_protocol(input_file, output_file):
    """Converte arquivo de protocolo para formato simples."""
    if not os.path.exists(input_file):
        print(f"Arquivo não encontrado: {input_file}")
        return False
    
    # Criar diretório de saída se necessário
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    genuine_count = 0
    spoof_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            parts = line.strip().split()
            
            if len(parts) < 2:
                continue
            
            # ASVspoof 2019 formato típico:
            # PA: SPEAKER_ID FILENAME - - bonafide/spoof
            # LA: SPEAKER_ID FILENAME - SYSTEM bonafide/spoof
            
            # O nome do arquivo geralmente está na segunda coluna
            file_id = parts[1]
            
            # O label está na última coluna
            label_str = parts[-1].lower()
            
            if label_str in ['bonafide', 'genuine']:
                label = 'genuine'
                genuine_count += 1
            elif label_str == 'spoof':
                label = 'spoof'
                spoof_count += 1
            else:
                print(f"Aviso: Label desconhecido na linha {line_num}: {label_str}")
                continue
            
            # Escrever no formato simples - CORREÇÃO: file_id primeiro, depois label
            f_out.write(f"{file_id} {label}\n")
            converted_count += 1
    
    print(f"Conversão concluída: {output_file}")
    print(f"  Total convertido: {converted_count}")
    print(f"  Genuine: {genuine_count}")
    print(f"  Spoof: {spoof_count}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Converter protocolos ASVspoof")
    
    parser.add_argument('--dataset', type=str, default='PA', choices=['PA', 'LA'],
                        help='Dataset a converter')
    parser.add_argument('--output-dir', type=str, default='labels',
                        help='Diretório de saída para labels')
    
    args = parser.parse_args()
    
    # Definir caminhos baseado no dataset
    if args.dataset == 'PA':
        base_dir = "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols"
        files = {
            'train': 'ASVspoof2019.PA.cm.train.trl.txt',
            'dev': 'ASVspoof2019.PA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.PA.cm.eval.trl.txt'
        }
    else:  # LA
        base_dir = "E:/ASV 2019 DATA/LA/ASVspoof2019_LA_cm_protocols"
        files = {
            'train': 'ASVspoof2019.LA.cm.train.trl.txt',
            'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
            'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'
        }
    
    # Converter cada arquivo
    for split, filename in files.items():
        input_path = os.path.join(base_dir, filename)
        output_path = os.path.join(args.output_dir, args.dataset, f"{split}_labels.txt")
        
        print(f"\nConvertendo {split}...")
        convert_protocol(input_path, output_path)


if __name__ == "__main__":
    main()