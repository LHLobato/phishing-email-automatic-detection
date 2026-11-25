#!/usr/bin/env bash
set -euo pipefail

# Diretório onde os melhores modelos e métricas serão salvos
SAVE_DIR="${SAVE_DIR:-./saved_models}"
USE_LORA="${USE_LORA:-true}"

# Bases de dados comentadas na linha de carregamento do kagglehub
DATASETS=(
    "CEAS_08.csv"
    "Nazario.csv"
    "SpamAssasin.csv"
    "Nigerian_Fraud.csv"
)

# 4 modelos presentes no script original
MODELS=(
    "BERT-Base"
    "ModernBERT"
    "RoBERTa"
    "DistilBERT"
)

# Seeds solicitadas
SEEDS=(0 100 1000)

echo "=========================================================="
echo "Iniciando automação de treinamento"
echo "Diretório de salvamento: ${SAVE_DIR}"
echo "Uso de LoRA: ${USE_LORA}"
echo "Bases de dados: ${DATASETS[*]}"
echo "Modelos: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================================="

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo ""
            echo "----------------------------------------------------------"
            echo "[EXECUÇÃO] Base: ${dataset} | Modelo: ${model} | Seed: ${seed} | LoRA: ${USE_LORA}"
            echo "----------------------------------------------------------"
            python3 training.py \
                --dataset "${dataset}" \
                --model "${model}" \
                --use_lora "${USE_LORA}" \
                --save_dir "${SAVE_DIR}" \
                --seeds "${seed}"
        done
    done
done

echo ""
echo "=========================================================="
echo "Treinamento automatizado finalizado com sucesso!"
echo "Os melhores modelos foram salvos em: ${SAVE_DIR}/best_models"
echo "=========================================================="
