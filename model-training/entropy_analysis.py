import argparse
import math
import os
from collections import Counter
import kagglehub
from kagglehub import KaggleDatasetAdapter
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def calculate_shannon_entropy(text: str) -> float:
    """Calcula a entropia de Shannon (em bits por caractere) de uma string."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def load_dataset_df(dataset_arg: str) -> pd.DataFrame:
    if os.path.isfile(dataset_arg):
        print(f"Carregando base de dados local: {dataset_arg}")
        return pd.read_csv(dataset_arg)
    print(f"Carregando base de dados via KaggleHub: {dataset_arg}")
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "naserabdullahalam/phishing-email-dataset",
        dataset_arg,
    )


def analyze_dataset_entropy(dataset_arg: str, output_dir: str = None) -> dict:
    df = load_dataset_df(dataset_arg)
    dataset_name = os.path.splitext(os.path.basename(dataset_arg))[0]

    if 'body' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"A base {dataset_arg} deve conter as colunas 'body' e 'label'. Colunas encontradas: {list(df.columns)}")

    data_df = df[['body', 'label']].dropna(subset=['body', 'label']).copy()
    data_df['body'] = data_df['body'].astype(str)
    data_df['label'] = data_df['label'].astype(int)

    # Calcula entropia por e-mail
    data_df['entropy'] = data_df['body'].apply(calculate_shannon_entropy)

    legit_entropy = data_df[data_df['label'] == 0]['entropy']
    malicious_entropy = data_df[data_df['label'] == 1]['entropy']

    # Estatísticas descritivas por classe
    stats_legit = {
        "count": int(len(legit_entropy)),
        "mean": float(legit_entropy.mean()),
        "std": float(legit_entropy.std()),
        "median": float(legit_entropy.median()),
        "iqr": float(legit_entropy.quantile(0.75) - legit_entropy.quantile(0.25)),
        "min": float(legit_entropy.min()),
        "max": float(legit_entropy.max()),
    }

    stats_malicious = {
        "count": int(len(malicious_entropy)),
        "mean": float(malicious_entropy.mean()),
        "std": float(malicious_entropy.std()),
        "median": float(malicious_entropy.median()),
        "iqr": float(malicious_entropy.quantile(0.75) - malicious_entropy.quantile(0.25)),
        "min": float(malicious_entropy.min()),
        "max": float(malicious_entropy.max()),
    }

    # Relação com as classes (Correlação Ponto-Bisserial, Spearman, Testes de Hipótese e ROC-AUC)
    pb_corr, pb_pval = stats.pointbiserialr(data_df['label'], data_df['entropy'])
    spearman_corr, spearman_pval = stats.spearmanr(data_df['label'], data_df['entropy'])
    t_stat, t_pval = stats.ttest_ind(malicious_entropy, legit_entropy, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(malicious_entropy, legit_entropy, alternative='two-sided')
    roc_auc = roc_auc_score(data_df['label'], data_df['entropy'])

    print(f"\n============================================================")
    print(f"Análise de Entropia por E-mail - Base: {dataset_name}")
    print(f"============================================================")
    print(f"Total de e-mails analisados: {len(data_df)}")
    print(f"\n--- Classe Legítima (0) [n = {stats_legit['count']}] ---")
    print(f"  Média ± Desvio Padrão : {stats_legit['mean']:.4f} ± {stats_legit['std']:.4f}")
    print(f"  Mediana (IQR)         : {stats_legit['median']:.4f} ({stats_legit['iqr']:.4f})")
    print(f"  Mínimo / Máximo       : {stats_legit['min']:.4f} / {stats_legit['max']:.4f}")

    print(f"\n--- Classe Maliciosa (1) [n = {stats_malicious['count']}] ---")
    print(f"  Média ± Desvio Padrão : {stats_malicious['mean']:.4f} ± {stats_malicious['std']:.4f}")
    print(f"  Mediana (IQR)         : {stats_malicious['median']:.4f} ({stats_malicious['iqr']:.4f})")
    print(f"  Mínimo / Máximo       : {stats_malicious['min']:.4f} / {stats_malicious['max']:.4f}")

    print(f"\n--- Relação entre Entropia e Classe (1 = Maliciosa vs 0 = Legítima) ---")
    print(f"  Diferença de Médias (Maliciosa - Legítima) : {stats_malicious['mean'] - stats_legit['mean']:+.4f} bits/char")
    print(f"  Correlação Ponto-Bisserial (r_pb)          : {pb_corr:.4f} (p-valor = {pb_pval:.4e})")
    print(f"  Correlação de Spearman (rho)               : {spearman_corr:.4f} (p-valor = {spearman_pval:.4e})")
    print(f"  Teste t de Welch (t-stat)                  : {t_stat:.4f} (p-valor = {t_pval:.4e})")
    print(f"  Teste Mann-Whitney U                       : {u_stat:.4f} (p-valor = {u_pval:.4e})")
    print(f"  ROC-AUC usando apenas Entropia             : {roc_auc:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"entropy_per_email_{dataset_name}.csv")
        data_df[['label', 'entropy']].to_csv(out_csv, index=False)
        print(f"\nEntropia por e-mail salva em: {out_csv}")

    return {
        "dataset": dataset_name,
        "total_emails": len(data_df),
        "legit_count": stats_legit["count"],
        "legit_mean": stats_legit["mean"],
        "legit_std": stats_legit["std"],
        "legit_median": stats_legit["median"],
        "malicious_count": stats_malicious["count"],
        "malicious_mean": stats_malicious["mean"],
        "malicious_std": stats_malicious["std"],
        "malicious_median": stats_malicious["median"],
        "mean_diff_malicious_minus_legit": stats_malicious["mean"] - stats_legit["mean"],
        "point_biserial_r": pb_corr,
        "point_biserial_p": pb_pval,
        "spearman_rho": spearman_corr,
        "spearman_p": spearman_pval,
        "welch_t_stat": t_stat,
        "welch_t_p": t_pval,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_pval,
        "roc_auc_entropy": roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calcula a entropia de Shannon por e-mail e verifica sua relação com as classes legítima (0) e maliciosa (1)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["CEAS_08.csv", "Nazario.csv", "SpamAssasin.csv", "Nigerian_Fraud.csv"],
        help="Lista de arquivos .csv (do KaggleHub ou locais) para analisar."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./entropy_analysis",
        help="Diretório para salvar os resultados da análise de entropia."
    )
    args = parser.parse_args()

    summary_rows = []
    for ds in args.datasets:
        res = analyze_dataset_entropy(ds, args.output_dir)
        summary_rows.append(res)

    if len(summary_rows) > 0 and args.output_dir:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.output_dir, "entropy_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nResumo consolidado da análise de entropia salvo em: {summary_path}")


if __name__ == "__main__":
    main()
