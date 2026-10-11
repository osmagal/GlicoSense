"""
GlicoSense - Pipeline de Treinamento e Seleção de Modelos de Risco de Diabetes
Autor: Equipe GlicoSense
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo headless para salvar figuras sem necessidade de servidor gráfico
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

def main():
    print("=" * 70)
    print("           GLICOSENSE - TREINAMENTO E COMPARAÇÃO DE MODELOS")
    print("=" * 70)

    # 1. Carregamento dos dados
    file_path = "diabetes_risk.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de dados '{file_path}' não encontrado.")

    df = pd.read_csv(file_path)
    print(f"\n[1] Dataset carregado com sucesso!")
    print(f"    Dimensões originais: {df.shape}")
    print(f"    Linhas duplicadas identificadas: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"    Dimensões após remoção de duplicatas: {df.shape}")

    # 2. Pré-processamento e separação de variáveis
    # Remoção do identificador artificial (patient_id) para evitar viés/leakage
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])
        print("    Coluna 'patient_id' descartada por ser identificador não preditivo.")

    target_column = "diabetes_risk"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    num_cols = X.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print(f"\n[2] Features identificadas:")
    print(f"    Numéricas ({len(num_cols)}): {num_cols}")
    print(f"    Categóricas ({len(cat_cols)}): {cat_cols}")
    print(f"    Distribuição da variável alvo '{target_column}':")
    for cls_name, count in y.value_counts().items():
        pct = (count / len(y)) * 100
        print(f"      - {cls_name}: {count} ({pct:.1f}%)")

    # 3. Divisão Treino e Teste (Estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[3] Divisão dos Dados:")
    print(f"    Conjunto de Treinamento: {X_train.shape[0]} amostras")
    print(f"    Conjunto de Teste:       {X_test.shape[0]} amostras")

    # 4. Construção de Pipelines de Transformação
    # Pipeline com padronização (StandardScaler) para modelos sensíveis a escala
    prep_scaled = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # Pipeline sem padronização para modelos baseados em árvores
    prep_unscaled = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # 5. Definição dos Modelos Candidatos
    models = {
        "Naive Bayes": Pipeline([("prep", prep_scaled), ("clf", GaussianNB())]),
        "Decision Tree": Pipeline([("prep", prep_unscaled), ("clf", DecisionTreeClassifier(random_state=42, max_depth=5))]),
        "K-Nearest Neighbors (KNN)": Pipeline([("prep", prep_scaled), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Logistic Regression": Pipeline([("prep", prep_scaled), ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))]),
        "Random Forest": Pipeline([("prep", prep_unscaled), ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]),
        "HistGradientBoosting": Pipeline([("prep", prep_unscaled), ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=100, learning_rate=0.05, max_depth=3, min_samples_leaf=50))])
    }

    # 6. Treinamento e Avaliação com Validação Cruzada (5 Folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("\n[4] Treinando modelos e calculando validação cruzada 5-Fold...")
    for name, pipe in models.items():
        print(f"    -> Treinando {name}...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else np.nan
        cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
        cr = classification_report(y_test, y_pred, labels=pipe.classes_, output_dict=True)

        results.append({
            "Model": name,
            "CV_Accuracy_Mean": cv_scores.mean(),
            "CV_Accuracy_Std": cv_scores.std(),
            "Test_Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision_Macro": prec_macro,
            "Recall_Macro": rec_macro,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "ROC_AUC_OVR": auc,
            "Confusion_Matrix": cm,
            "Classes": list(pipe.classes_),
            "Classification_Report": cr,
            "Pipeline": pipe
        })

    # 7. Tabela Comparativa de Resultados
    comparison_table = pd.DataFrame([{
        "Modelo": r["Model"],
        "Acurácia Teste (%)": round(r["Test_Accuracy"] * 100, 2),
        "CV Acurácia 5-Fold (%)": f"{r['CV_Accuracy_Mean']*100:.2f} ± {r['CV_Accuracy_Std']*100:.2f}",
        "Acurácia Bal. (%)": round(r["Balanced_Accuracy"] * 100, 2),
        "F1 Macro (%)": round(r["F1_Macro"] * 100, 2),
        "F1 Weighted (%)": round(r["F1_Weighted"] * 100, 2),
        "ROC-AUC (OVR)": round(r["ROC_AUC_OVR"], 4)
    } for r in results])

    print("\n" + "=" * 80)
    print("                      TABELA COMPARATIVA DE MODELOS")
    print("=" * 80)
    print(comparison_table.to_string(index=False))

    # 8. Seleção do Melhor Modelo
    best_idx = np.argmax([r["Test_Accuracy"] for r in results])
    best_model_data = results[best_idx]
    best_name = best_model_data["Model"]
    best_pipeline = best_model_data["Pipeline"]

    print("\n" + "=" * 80)
    print(f"                      MELHOR MODELO ESCOLHIDO: {best_name.upper()}")
    print("=" * 80)
    print(f"  - Acurácia de Teste:     {best_model_data['Test_Accuracy'] * 100:.2f}%")
    print(f"  - Validação Cruzada:     {best_model_data['CV_Accuracy_Mean'] * 100:.2f}% (± {best_model_data['CV_Accuracy_Std'] * 100:.2f}%)")
    print(f"  - Acurácia Balanceada:   {best_model_data['Balanced_Accuracy'] * 100:.2f}%")
    print(f"  - F1-Score (Macro):      {best_model_data['F1_Macro'] * 100:.2f}%")
    print(f"  - F1-Score (Weighted):   {best_model_data['F1_Weighted'] * 100:.2f}%")
    print(f"  - ROC-AUC (OVR):         {best_model_data['ROC_AUC_OVR']:.4f}")

    print("\nRelatório de Classificação Detalhado do Melhor Modelo:")
    print(classification_report(y_test, best_pipeline.predict(X_test), labels=best_pipeline.classes_))

    # 9. Salvando o Arquivo Resultante do Melhor Modelo (.joblib)
    output_model_path = "melhor_modelo.joblib"
    joblib.dump(best_pipeline, output_model_path)
    print(f"\n[+] Pipeline completa do melhor modelo salva com sucesso em: '{output_model_path}'")

    # 10. Geração e Salvamento de Gráficos
    # Gráfico 1: Comparativo de Acurácia
    plt.figure(figsize=(10, 5))
    acc_values = [r["Test_Accuracy"] * 100 for r in results]
    model_names = [r["Model"] for r in results]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]
    bars = plt.bar(model_names, acc_values, color=colors)
    plt.title("Comparação da Acurácia de Teste dos Modelos", fontsize=13, fontweight="bold")
    plt.xlabel("Modelo", fontsize=11)
    plt.ylabel("Acurácia de Teste (%)", fontsize=11)
    plt.ylim(60, 85)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.3, f"{height:.2f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("comparacao_modelos.png", dpi=300)
    plt.close()
    print("[+] Gráfico comparativo gerado em: 'comparacao_modelos.png'")

    # Gráfico 2: Matriz de Confusão do Melhor Modelo
    plt.figure(figsize=(7, 6))
    cm = best_model_data["Confusion_Matrix"]
    classes = best_model_data["Classes"]
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {best_name}", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight="bold")
    plt.ylabel("Classe Real", fontsize=11)
    plt.xlabel("Classe Predita", fontsize=11)
    plt.tight_layout()
    plt.savefig("matriz_confusao_melhor_modelo.png", dpi=300)
    plt.close()
    print("[+] Matriz de confusão gerada em: 'matriz_confusao_melhor_modelo.png'")

    print("\nExecução concluída com sucesso!")

if __name__ == "__main__":
    main()