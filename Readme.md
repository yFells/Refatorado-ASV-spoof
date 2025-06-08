Sistema de Detecção de Ataques Spoofing de Replay (Versão Simplificada)
Este é um sistema simplificado para detecção de ataques de replay em ASVspoof 2019, configurado para usar apenas 30% dos dados mantendo a proporção entre amostras genuínas e spoofing.
📋 Arquivos do Sistema

feature_extraction_fix.py - Extração de características com amostragem proporcional
simple_model.py - Modelo CNN simplificado
train_simple.py - Script de treinamento
test_simple.py - Script de teste e avaliação
run_pipeline.py - Script principal para executar todo o pipeline
requirements.txt - Dependências do projeto

🚀 Instalação

Instalar dependências:

bashpip install -r requirements.txt

Verificar estrutura de diretórios:
Certifique-se de que os datasets estão organizados conforme esperado:


PA: E:\ASV 2019 DATA\PA\
LA: E:\ASV 2019 DATA\LA\

💻 Uso Rápido
Executar pipeline completo (recomendado):
bash# Para dataset PA (Physical Access - Replay attacks)
python run_pipeline.py --dataset PA

# Para dataset LA (Logical Access - Synthesis/conversion)
python run_pipeline.py --dataset LA

Executar etapas individuais:

Extração de características (30% dos dados):

bashpython feature_extraction_fix.py \
    --train-audio-dir "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_train/flac" \
    --dev-audio-dir "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_dev/flac" \
    --eval-audio-dir "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_eval/flac" \
    --output-dir "output/features" \
    --train-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt" \
    --dev-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt" \
    --eval-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt" \
    --sample-proportion 0.3

Treinamento:

bashpython train_simple.py \
    --train-features-dir "output/features/train" \
    --dev-features-dir "output/features/dev" \
    --train-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trl.txt" \
    --dev-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt" \
    --num-epochs 30

Teste:

bashpython test_simple.py \
    --test-features-dir "output/features/eval" \
    --test-labels-file "E:/ASV 2019 DATA/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt" \
    --checkpoint "checkpoints/best_model.pth"
📊 Características do Sistema

Amostragem: Usa 30% dos dados mantendo proporção genuíno/spoof
Modelo: CNN simplificada com 3 camadas convolucionais
Características: MFCC + CQCC (180 dimensões)
Métricas: EER, AUC, Acurácia
Tempo estimado: ~30-60 minutos para pipeline completo

📈 Resultados Esperados
Com 30% dos dados, você deve obter:

EER: ~5-10% (dependendo do dataset)
AUC: >0.90
Tempo de treinamento: ~20-30 minutos

🔧 Personalização
Para ajustar a proporção de dados, modifique o parâmetro --sample-proportion:

0.1 = 10% dos dados
0.3 = 30% dos dados (padrão)
0.5 = 50% dos dados
1.0 = 100% dos dados

📝 Notas

O sistema mantém automaticamente a proporção entre amostras genuínas e spoof
Os resultados são salvos em formato JSON e PNG
Checkpoints são salvos a cada época melhor que a anterior
Use GPU se disponível para acelerar o treinamento

🐛 Solução de Problemas

Erro ao carregar áudio: Instale ffmpeg
Memória insuficiente: Reduza --batch-size
Treinamento lento: Reduza --sample-proportion ou --num-epochs
