import argparse
import os
import random
import json
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_recall_curve, auc
import evaluate
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd


MODEL_CONFIGS = {
    "BERT-Base": "bert-base-uncased",
    "ModernBERT": "answerdotai/ModernBERT-base",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased"
}


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_dataset_df(dataset_arg: str) -> pd.DataFrame:
    if os.path.isfile(dataset_arg):
        print(f"Carregando base de dados local: {dataset_arg}")
        return pd.read_csv(dataset_arg)
    print(f"Carregando base de dados via KaggleHub: {dataset_arg}")
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "naserabdullahalam/phishing-email-dataset",
        dataset_arg,  # Ex: "CEAS_08.csv", "Nazario.csv", "SpamAssasin.csv", "Nigerian_Fraud.csv"
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Estabilização numérica para softmax
    shifted_logits = predictions - np.max(predictions, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]

    preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    # Calcular PR-AUC
    precisions_curve, recalls_curve, _ = precision_recall_curve(labels, pos_probs)
    pr_auc = auc(recalls_curve, precisions_curve)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'sim', 's'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'nao', 'não'):
        return False
    else:
        raise argparse.ArgumentTypeError('Valor booleano esperado (true/false).')


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento automatizado de modelos para detecção de phishing.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CEAS_08.csv",
        help="Nome do arquivo .csv no dataset do KaggleHub (ex: CEAS_08.csv, Nazario.csv, SpamAssasin.csv, Nigerian_Fraud.csv) ou caminho local para um arquivo .csv."
    )
    parser.add_argument(
        "--model",
        type=str,
         nargs="+",
        default=list(MODEL_CONFIGS.keys()),
        help=f"Modelo(s) a serem treinados. Opções mapeadas: {list(MODEL_CONFIGS.keys())} ou identificador HuggingFace direto."
    )
    parser.add_argument(
        "--use_lora",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Define se utiliza LoRA no treinamento (True/False). Padrão: True."
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Flag alternativa para desativar o uso de LoRA."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Diretório base para salvamento dos melhores modelos e resultados."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Lista de seeds para execução do treinamento (ex: --seeds 0 100 1000)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Número máximo de épocas de treinamento (padrão: 20)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Tamanho do batch de treino e avaliação (padrão: 16)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_lora = False if args.no_lora else args.use_lora

    os.makedirs(args.save_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

    df = load_dataset_df(args.dataset)

    # Filtrando colunas relevantes e removendo valores nulos
    data_df = df[['body', 'label']].dropna(subset=['body', 'label']).copy()
    data_df['body'] = data_df['body'].astype(str)
    data_df['label'] = data_df['label'].astype(int)
    data_df = data_df.rename(columns={'body': 'text', 'label': 'label'})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()

    all_results = []

    for seed in args.seeds:
        print(f"\n==========================================")
        print(f"Iniciando rodada com SEED: {seed} | Base: {dataset_name} | LoRA: {use_lora}")
        print(f"==========================================")
        set_all_seeds(seed)

        # Divisão 70 / 15 / 15
        train_df, temp_df = train_test_split(
            data_df, test_size=0.30, random_state=seed, stratify=data_df['label']
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=seed, stratify=temp_df['label']
        )

        print(f"Treino: {len(train_df)} | Validação: {len(val_df)} | Teste: {len(test_df)}")

        train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

        for model_key in args.model:
            model_path = MODEL_CONFIGS.get(model_key, model_key)
            safe_model_name = model_key.replace("/", "_")
            lora_tag = "lora" if use_lora else "full"

            run_id = f"{dataset_name}_{safe_model_name}_{lora_tag}_seed{seed}"
            checkpoints_dir = os.path.join(args.save_dir, "checkpoints", run_id)
            best_model_dir = os.path.join(args.save_dir, "best_models", dataset_name, safe_model_name, f"{lora_tag}_seed_{seed}")
            os.makedirs(best_model_dir, exist_ok=True)

            print(f"\n=== Iniciando o treinamento do modelo: {model_key} ({model_path}) | Seed: {seed} ===")

            # Inicializa Tokenizador
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, max_length=500, padding='max_length')

            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_val = val_dataset.map(tokenize_function, batched=True)
            tokenized_test = test_dataset.map(tokenize_function, batched=True)

            # Inicializa o Modelo
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id

            if use_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none"
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

            model = model.to(device)

            # Argumentos de Treinamento
            # load_best_model_at_end=True com metric_for_best_model="eval_loss" garante que
            # ao final do treino o modelo carregado em memória é o de menor perda de validação.
            training_args = TrainingArguments(
                output_dir=checkpoints_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                learning_rate=2e-4 if use_lora else 2e-5,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.epochs,
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_steps=100,
                disable_tqdm=False,
                report_to="none",
                fp16=use_fp16,
                seed=seed,
                data_seed=seed
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
            )

            # Treinar
            trainer.train()

            # Salvar o melhor modelo (menor perda de validação) desta rodada de treino
            best_val_loss = trainer.state.best_metric
            print(f"\n--- Salvando o melhor modelo (menor eval_loss = {best_val_loss}) em: {best_model_dir} ---")
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # Avaliar no conjunto de teste
            print(f"\n--- Avaliando {model_key} (Seed {seed}) no conjunto de Teste ---")
            test_metrics = trainer.evaluate(tokenized_test)
            test_metrics["model"] = model_key
            test_metrics["dataset"] = dataset_name
            test_metrics["seed"] = seed
            test_metrics["use_lora"] = use_lora
            test_metrics["best_val_loss"] = best_val_loss
            test_metrics["saved_model_dir"] = best_model_dir

            # Salvar métricas individuais junto ao modelo
            metrics_path = os.path.join(best_model_dir, "test_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=4)

            all_results.append(test_metrics)
            print(test_metrics)

    # Exibir e salvar resumo das métricas no final
    summary_df = pd.DataFrame(all_results)
    print("\n=== Resumo Final das Métricas ===")
    print(summary_df.to_string(index=False))

    summary_csv_path = os.path.join(args.save_dir, f"summary_results_{dataset_name}.csv")
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nResumo salvo em: {summary_csv_path}")


if __name__ == "__main__":
    main()
