#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo simplificado para extração de características acústicas com suporte a amostragem proporcional.
"""

import numpy as np
import os
import librosa
import soundfile as sf
from scipy.signal import lfilter
from tqdm import tqdm
import argparse
import random
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Classe simplificada para extração de características acústicas."""
    
    def __init__(self, sample_rate=16000, n_mfcc=30, n_cqcc=30, n_mels=257,
                window_size=0.025, hop_size=0.010, pre_emphasis=0.97):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_cqcc = n_cqcc
        self.n_mels = n_mels
        self.window_size = window_size
        self.hop_size = hop_size
        self.pre_emphasis = pre_emphasis
        
        # Parâmetros para FFT
        self.n_fft = int(2 ** np.ceil(np.log2(self.window_size * self.sample_rate)))
        self.hop_length = int(self.hop_size * self.sample_rate)
        self.win_length = int(self.window_size * self.sample_rate)

    def load_audio(self, audio_path):
        """Carrega arquivo de áudio."""
        try:
            # Tentar com soundfile primeiro
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio, self.sample_rate
        except:
            try:
                # Fallback para librosa
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                return audio, sr
            except:
                print(f"Erro ao carregar {audio_path}")
                return None, None
        
    def preprocess_audio(self, audio):
        """Pré-processamento do áudio."""
        if len(audio) == 0:
            return audio
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        return lfilter([1, -self.pre_emphasis], [1], audio)
        
    def extract_mfcc(self, audio):
        """Extrai MFCC."""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window='hamming'
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    
    def extract_cqcc(self, audio):
        """Extrai CQCC simplificado."""
        # Versão simplificada - usar MFCC como proxy
        return self.extract_mfcc(audio)
    
    def extract_mel_spectrogram(self, audio):
        """Extrai espectrograma Mel."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            window='hamming', n_mels=self.n_mels
        )
        return librosa.power_to_db(mel_spec)
    
    def extract_features(self, audio_path):
        """Extrai todas as características de um arquivo."""
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return None
            
        audio = self.preprocess_audio(audio)
        
        features = {
            'mfcc': self.extract_mfcc(audio),
            'cqcc': self.extract_cqcc(audio),
            'mel_spectrogram': self.extract_mel_spectrogram(audio),
            'lbp': np.zeros((257, 100)),  # Placeholder
            'glcm': np.zeros(20),  # Placeholder
            'lpq': np.zeros((257, 100))  # Placeholder
        }
        
        return features
    
    def batch_feature_extraction(self, audio_dir, output_dir, labels_file, sample_proportion=0.3):
        """Extrai características com amostragem proporcional."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Encontrar todos os arquivos
        all_audio_files = []
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file.endswith('.flac'):
                    all_audio_files.append(os.path.join(root, file))
        
        # Carregar labels
        file_labels = {}
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        file_id = parts[0]
                        label = 'genuine' if parts[-1].lower() in ['bonafide', 'genuine'] else 'spoof'
                        file_labels[file_id] = label
        
        # Separar por classe
        genuine_files = []
        spoof_files = []
        
        for audio_path in all_audio_files:
            file_id = os.path.splitext(os.path.basename(audio_path))[0]
            if file_id in file_labels:
                if file_labels[file_id] == 'genuine':
                    genuine_files.append(audio_path)
                else:
                    spoof_files.append(audio_path)
        
        # Amostrar proporcionalmente
        n_genuine = int(len(genuine_files) * sample_proportion)
        n_spoof = int(len(spoof_files) * sample_proportion)
        
        sampled_genuine = random.sample(genuine_files, n_genuine) if n_genuine > 0 else []
        sampled_spoof = random.sample(spoof_files, n_spoof) if n_spoof > 0 else []
        
        files_to_process = sampled_genuine + sampled_spoof
        random.shuffle(files_to_process)
        
        print(f"Processando {len(files_to_process)} arquivos ({n_genuine} genuínos, {n_spoof} spoof)")
        
        # Processar arquivos
        for audio_path in tqdm(files_to_process, desc="Extraindo características"):
            try:
                filename = os.path.basename(audio_path)
                file_id = os.path.splitext(filename)[0]
                
                features = self.extract_features(audio_path)
                if features is not None:
                    output_path = os.path.join(output_dir, f"{file_id}.npz")
                    np.savez(output_path, **features)
            except Exception as e:
                print(f"Erro ao processar {audio_path}: {str(e)}")
        
        print(f"Extração concluída. Características salvas em: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extração de características com amostragem")
    
    parser.add_argument('--train-audio-dir', type=str, help='Diretório de áudio de treino')
    parser.add_argument('--dev-audio-dir', type=str, help='Diretório de áudio de validação')
    parser.add_argument('--eval-audio-dir', type=str, help='Diretório de áudio de teste')
    parser.add_argument('--output-dir', type=str, required=True, help='Diretório de saída')
    parser.add_argument('--train-labels-file', type=str, help='Arquivo de labels de treino')
    parser.add_argument('--dev-labels-file', type=str, help='Arquivo de labels de validação')
    parser.add_argument('--eval-labels-file', type=str, help='Arquivo de labels de teste')
    parser.add_argument('--sample-proportion', type=float, default=0.3, help='Proporção de amostragem')
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor()
    
    # Processar cada conjunto
    if args.train_audio_dir and args.train_labels_file:
        train_output = os.path.join(args.output_dir, 'train')
        print(f"\nExtraindo características de treino com {args.sample_proportion*100}% dos dados...")
        extractor.batch_feature_extraction(
            args.train_audio_dir, train_output, 
            args.train_labels_file, args.sample_proportion
        )
    
    if args.dev_audio_dir and args.dev_labels_file:
        dev_output = os.path.join(args.output_dir, 'dev')
        print(f"\nExtraindo características de validação com {args.sample_proportion*100}% dos dados...")
        extractor.batch_feature_extraction(
            args.dev_audio_dir, dev_output,
            args.dev_labels_file, args.sample_proportion
        )
    
    if args.eval_audio_dir and args.eval_labels_file:
        eval_output = os.path.join(args.output_dir, 'eval')
        print(f"\nExtraindo características de teste com {args.sample_proportion*100}% dos dados...")
        extractor.batch_feature_extraction(
            args.eval_audio_dir, eval_output,
            args.eval_labels_file, args.sample_proportion
        )


if __name__ == "__main__":
    main()
