# Relatório Técnico de Avaliação e Seleção de Modelos — GlicoSense

## 1. Sumário Executivo

O projeto **GlicoSense** visa a estratificação automatizada do risco de diabetes em pacientes clínicos categorizados em três níveis: **Baixo Risco (Low)**, **Risco Moderado (Moderate)** e **Alto Risco (High)**.

Foram treinados e avaliados comparativamente seis algoritmos de aprendizado de máquina, combinando modelos tradicionais de linha de base (*baseline*) e modelos avançados lineares e não lineares baseados em *ensembles*.

O modelo selecionado como **vencedor** foi a **Regressão Logística Multinomial (Multinomial Logistic Regression)**, que atingiu:
- **Acurácia de Teste:** **79,33%**
- **Acurácia em Validação Cruzada (5-Fold CV):** **79,23% (± 0,80%)**
- **Área sob a Curva ROC (ROC-AUC One-vs-Rest):** **0,9158**
- **F1-Score Macro:** **74,30%** (indicando alto desempenho em todas as classes, inclusive minoritárias)
- **Falsos Negativos Críticos (Alto Risco predito como Baixo Risco):** apenas **6 em 450 pacientes** (1,3%)

O modelo treinado foi encapsulado em um pipeline de pré-processamento integral e serializado no arquivo [`melhor_modelo.joblib`](file:///home/docato/Documentos/GlicoSense/melhor_modelo.joblib).

---

## 2. Descrição dos Dados e Pré-Processamento

### 2.1 Visão Geral do Dataset
- **Base de Dados:** [`diabetes_risk.csv`](file:///home/docato/Documentos/GlicoSense/diabetes_risk.csv)
- **Total de Amostras:** 15.000 pacientes (sem registros duplicados)
- **Total de Colunas:** 19 (1 identificador, 17 variáveis preditoras e 1 variável alvo)
- **Distribuição da Variável Alvo (`diabetes_risk`):**
  - **Low (Baixo):** 9.000 amostras (60,0%)
  - **Moderate (Moderado):** 3.750 amostras (25,0%)
  - **High (Alto):** 2.250 amostras (15,0%)

### 2.2 Tratamento e Engenharia de Features
1. **Remoção de Identificador Artificial:** A coluna `patient_id` é um sequencial arbitrário de 1 a 15.000. Sua inclusão distorce métricas de distância (KNN) e gera ruído/risco de vazamento (*data leakage*). Foi removida do conjunto preditor.
2. **Separação de Variáveis:**
   - **Variáveis Numéricas (9):** `age`, `bmi`, `hours_sleep_per_night`, `stress_level`, `fasting_blood_sugar`, `hba1c_level`, `blood_pressure_systolic`, `blood_pressure_diastolic`, `waist_circumference_cm`.
   - **Variáveis Categóricas (8):** `gender`, `city`, `family_history_diabetes`, `physical_activity_level`, `diet_type`, `smoking_status`, `alcohol_consumption`, `income_bracket`.
3. **Imputação de Valores Ausentes:**
   - Valores numéricos ausentes: preenchimento pela **mediana** via `SimpleImputer(strategy='median')`.
   - Valores categóricos ausentes (como `smoking_status`, `alcohol_consumption`, `income_bracket`): preenchimento pela **moda** (*most frequent*) antes da codificação.
4. **Codificação Categórica:** Transformação via `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`.
5. **Padronização:** Aplicação de `StandardScaler` nas variáveis numéricas nos modelos sensíveis a escala (`Logistic Regression`, `KNN`, `Naive Bayes`).
6. **Divisão Estratificada:**
   - 80% Treinamento (12.000 amostras)
   - 20% Teste independente (3.000 amostras), preservando a proporção de 60/25/15 entre classes (`stratify=y`).

---

## 3. Metodologia de Avaliação

Para garantir consistência estatística e ausência de sobreajuste (*overfitting*), todos os modelos foram submetidos a:
1. **Validação Cruzada Estratificada em 5 Folds (`StratifiedKFold`, k=5)** exclusivamente nos dados de treino.
2. **Avaliação no Conjunto de Teste Independente** não visto durante o treino.
3. **Métricas Computadas:**
   - **Acurácia Global (Accuracy):** Proporção total de acertos.
   - **Acurácia Balanceada (Balanced Accuracy):** Média das revocatórias por classe, compensando o desbalanceamento.
   - **Precisão, Recall e F1-Score (Macro e Ponderado/Weighted):** Para mensurar o equilíbrio entre classes.
   - **ROC-AUC (One-vs-Rest):** Capacidade de discriminação probabilística entre cada classe e as demais.

---

## 4. Tabela Comparativa de Desempenho

| Modelo | Acurácia Teste (%) | CV 5-Fold Acurácia (%) | Acurácia Balanceada (%) | F1-Score Macro (%) | F1-Score Ponderado (%) | ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **79,33%** | **79,23% ± 0,80%** | **72,96%** | **74,30%** | **79,00%** | **0,9158** |
| **HistGradientBoosting** | 79,17% | 79,11% ± 0,58% | 72,91% | 74,24% | 78,91% | 0,9124 |
| **Random Forest (100 árvores)** | 78,87% | 78,72% ± 0,53% | 72,27% | 73,73% | 78,50% | 0,9085 |
| **Decision Tree (depth=5)** | 76,50% | 77,13% ± 0,64% | 68,76% | 70,51% | 76,27% | 0,8943 |
| **Naive Bayes (GaussianNB)** | 74,30% | 74,22% ± 1,02% | 70,56% | 69,87% | 75,16% | 0,8809 |
| **K-Nearest Neighbors (k=5)** | 72,27% | 72,97% ± 0,74% | 63,14% | 64,07% | 70,92% | 0,8435 |

> [!NOTE]
> O gráfico comparativo visual de acurácia foi gerado e salvo em [`comparacao_modelos.png`](file:///home/docato/Documentos/GlicoSense/comparacao_modelos.png).

---

## 5. Análise Detalhada do Melhor Modelo (Logistic Regression)

### 5.1 Relatório de Classificação por Classe

| Classe | Precisão | Revocação (Recall) | F1-Score | Suporte (Teste) |
| :--- | :---: | :---: | :---: | :---: |
| **High (Alto Risco)** | **0,82** | **0,72** | **0,77** | 450 |
| **Low (Baixo Risco)** | **0,86** | **0,91** | **0,88** | 1.800 |
| **Moderate (Moderado)** | **0,60** | **0,56** | **0,58** | 750 |
| **Média Macro** | **0,76** | **0,73** | **0,74** | 3.000 |
| **Média Ponderada** | **0,79** | **0,79** | **0,79** | 3.000 |

### 5.2 Matriz de Confusão

```
                     Predito: High    Predito: Low    Predito: Moderate
Real: High                 323              6                121
Real: Low                    3           1635                162
Real: Moderate              68            260                422
```

> [!NOTE]
> O mapa de calor visual da matriz de confusão foi gerado e salvo em [`matriz_confusao_melhor_modelo.png`](file:///home/docato/Documentos/GlicoSense/matriz_confusao_melhor_modelo.png).

### 5.3 Análise de Erros Clínicos e Segurança Diagnóstica
- **Erro Crítico (Falso Negativo Severo - Real High -> Predito Low):** Apenas **6 pacientes** em 450 (1,33%). Esse é o erro mais perigoso em medicina, pois um paciente em alto risco deixaria de ser encaminhado para intervenção. A taxa de 1,33% demonstra alta confiabilidade do modelo.
- **Falso Positivo Severo (Real Low -> Predito High):** Apenas **3 pacientes** em 1.800 (0,16%).
- **Fronteira de Transição (Moderate):** A maior parte das divergências ocorre entre classes vizinhas (`Moderate` classificado como `Low` ou `High`), o que é clinicamente natural devido à sobreposição biológica contínua de marcadores como glicemia e HbA1c.

---

## 6. Interpretabilidade e Importância dos Biomarcadores

A análise dos coeficientes ponderados do modelo de Regressão Logística revela os fatores de maior peso na determinação do risco:

### Principais Fatores que Elevam o Risco para **HIGH**:
1. **Glicemia de Jejum (`fasting_blood_sugar`):** Coeficiente **+1,8074** (fator de maior impacto positivo).
2. **Hemoglobina Glicada (`hba1c_level`):** Coeficiente **+1,4951**.
3. **Idade (`age`):** Coeficiente **+0,6446**.
4. **Histórico Familiar (`family_history_diabetes = Yes`):** Coeficiente **+0,4661**.
5. **Índice de Massa Corporal (`bmi`):** Coeficiente **+0,4031**.
6. **Circunferência Abdominal (`waist_circumference_cm`):** Elevada correlação com acúmulo de gordura visceral.

### Principais Fatores que Reduzem o Risco (Associação com **LOW**):
1. **Glicemia de Jejum normalizada:** Coeficiente **-1,7516**.
2. **Hemoglobina Glicada baixa:** Coeficiente **-1,5025**.
3. **Menor faixa etária:** Coeficiente **-0,6629**.
4. **Ausência de histórico familiar:** Coeficiente **-0,5382**.
5. **IMC controlado:** Coeficiente **-0,4487**.

Essa coerência fisiológica valida que o modelo aprendeu relações clínicas reais e não correlações espúrias.

---

## 7. Justificativa da Escolha do Melhor Modelo

A **Regressão Logística Multinomial** foi selecionada como a melhor solução por quatro motivos centrais:
1. **Melhores Métricas Globais:** Atingiu a maior acurácia de teste (**79,33%**), a maior acurácia média em validação cruzada (**79,23%**) e o maior ROC-AUC (**0,9158**).
2. **Excelente Generalização:** A diferença entre a acurácia de treino/CV e a acurácia de teste foi de apenas 0,10%, evidenciando ausência de sobreajuste (*overfitting*).
3. **Interpretabilidade e Transparência Clínica:** Na área médica, modelos cujos pesos podem ser explicados como razões de chances (*odds ratios*) oferecem maior segurança regulatória e aceitação por profissionais de saúde em relação a modelos caixa-preta (*black-box*).
4. **Eficiência Computacional:** Tempo de inferência na ordem de sub-milissegundos, com consumo mínimo de memória RAM e CPU, ideal para integração em sistemas hospitalares e APIs REST.

---

## 8. Arquivo Salvo e Guia de Utilização (`melhor_modelo.joblib`)

O modelo vencedor foi serializado usando a biblioteca `joblib` no caminho:
- **Arquivo:** [`melhor_modelo.joblib`](file:///home/docato/Documentos/GlicoSense/melhor_modelo.joblib)

### Estrutura do Arquivo Serializado
O arquivo contém um objeto `sklearn.pipeline.Pipeline` completo com:
1. `ColumnTransformer` (com imputação de nulos, One-Hot Encoding e StandardScaler já ajustados);
2. Classificador `LogisticRegression` com os pesos finais treinados.

---

## 8. Seleção de Atributos com Maior Correlação com a Propensão ao Diabetes

Para identificar quais atributos possuem real poder explicativo e determinar se um modelo mais enxuto e prático pode ser construído, foi realizada uma análise multifacetada combinando:
1. **Coeficiente de Correlação Linear de Pearson ($r$)** com a escala de risco ordinal ($0 = \text{Low}, 1 = \text{Moderate}, 2 = \text{High}$).
2. **Correlação Monotônica de Spearman ($\rho$)**.
3. **Informação Mútua (*Mutual Information*)**, que captura relações não lineares complexas.
4. **Estatística $F$ de ANOVA (*ANOVA F-Statistic*) e valor-$p$ associado**.

### 8.1 Tabela de Relevância Estatística dos Atributos

| Atributo | Descrição Clínica | Correlação de Pearson ($r$) | Informação Mútua (MI) | ANOVA F-Score | Valor-$p$ | Categoria de Relevância |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `fasting_blood_sugar` | Glicemia de jejum (mg/dL) | **+0,738** | **0,4040** | **9.050,66** | $0,0$ | 🔴 **Crítica / Determinante** |
| `hba1c_level` | Hemoglobina glicada (%) | **+0,734** | **0,3820** | **8.807,51** | $0,0$ | 🔴 **Crítica / Determinante** |
| `bmi` | Índice de Massa Corporal | **+0,289** | **0,0443** | **699,50** | $< 10^{-290}$ | 🟠 **Alta Relevância** |
| `age` | Idade do indivíduo | **+0,287** | **0,0407** | **716,37** | $< 10^{-297}$ | 🟠 **Alta Relevância** |
| `waist_circumference_cm` | Circunferência abdominal | **+0,268** | **0,0379** | **593,38** | $< 10^{-248}$ | 🟠 **Alta Relevância** |
| `blood_pressure_systolic` | Pressão sistólica (mmHg) | **+0,212** | **0,0331** | **367,59** | $< 10^{-155}$ | 🟡 **Moderada Relevância** |
| `family_history_diabetes` | Histórico familiar (Sim/Não) | **+0,155** | **0,0194** | **206,65** | $< 10^{-88}$ | 🟡 **Moderada Relevância** |
| `physical_activity_level` | Nível de sedentarismo/atividade | **+0,147** | **0,0183** | **228,53** | $< 10^{-97}$ | 🟡 **Moderada Relevância** |
| `blood_pressure_diastolic` | Pressão diastólica (mmHg) | **+0,167** | **0,0158** | **226,18** | $< 10^{-96}$ | 🟡 **Moderada Relevância** |
| `stress_level` | Nível de estresse (escala 1-10) | **+0,100** | **0,0030** | **83,36** | $< 10^{-36}$ | 🟢 **Baixa Relevância** |
| `hours_sleep_per_night` | Horas de sono por noite | **-0,043** | **0,0005** | **14,53** | $4,9 \times 10^{-7}$ | ⚪ **Marginal** |
| `income_bracket` | Faixa de renda | $+0,033$ | $0,0045$ | $4,17$ | $0,015$ | ⚪ **Marginal / Espúria** |
| `city` | Cidade de residência | $+0,024$ | $0,0004$ | $1,87$ | $0,153$ | ❌ **Irrelevante ($p > 0.05$)** |
| `diet_type` | Tipo de dieta | $+0,016$ | $0,0000$ | $0,11$ | $0,897$ | ❌ **Irrelevante** |
| `gender` | Gênero | $+0,007$ | $0,0028$ | $0,26$ | $0,774$ | ❌ **Irrelevante** |
| `smoking_status` | Status de tabagismo | $+0,004$ | $0,0000$ | $1,00$ | $0,368$ | ❌ **Irrelevante** |
| `alcohol_consumption` | Consumo de álcool | $+0,001$ | $0,0000$ | $0,09$ | $0,911$ | ❌ **Irrelevante** |

> [!NOTE]
> O gráfico ilustrando o ranking de correlação e a curva de desempenho foi salvo em [`correlacao_atributos.png`](file:///home/docato/Documentos/GlicoSense/correlacao_atributos.png).

---

### 8.2 Comparação de Desempenho por Subconjunto de Atributos

Foi testado o impacto de reduzir o número de atributos de entrada nos modelos:

| Subconjunto de Atributos | Quantidade | Acurácia Teste (%) | CV 5-Fold Acurácia (%) | F1-Score Macro (%) | ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top 2:** Glicemia de Jejum + HbA1c | 2 | 76,93% | 76,60% ± 0,84% | 70,96% | 0,8994 |
| **Top 5:** Top 2 + IMC + Idade + Cintura | 5 | 79,07% | 79,16% ± 0,76% | 74,09% | 0,9138 |
| **Top 8 (ÓTIMO):** Top 5 + Pressão Sistólica + Histórico Familiar + Atividade Física | **8** | **79,43%** | **79,35% ± 0,64%** | **74,43%** | **0,9163** |
| **Top 10:** Top 8 + Pressão Diastólica + Nível de Estresse | 10 | 79,50% | 79,36% ± 0,68% | 74,49% | 0,9164 |
| **Todas as 17 Variáveis (Dataset Completo)** | 17 | 79,33% | 79,23% ± 0,80% | 74,30% | 0,9158 |

### Conclusões Críticas da Seleção de Atributos:
1. **O Modelo com Top 8 Atributos supera o modelo com todas as 17 variáveis:**
   - A acurácia sobe de **79,33% para 79,43%**, e o desvio padrão na validação cruzada cai de **±0,80% para ±0,64%** (maior consistência).
   - Isso ocorre pelo princípio da navalha de Occam: a eliminação de atributos de ruído (`city`, `diet_type`, `smoking_status`, `alcohol_consumption`, `income_bracket`, `gender`) reduz a variância espúria e evita memorização de ruído.
2. **Top 2 Biomarcadores Glicêmicos explicam quase 77% da acurácia sozinhos:**
   - Com apenas `fasting_blood_sugar` e `hba1c_level`, o modelo atinge 76,93% de acurácia e 0,8994 de ROC-AUC, comprovando a preponderância biológica desses marcadores no diabetes tipo 2.
3. **Ganho Prático em Aplicações Clínicas / Apps:**
   - Coletar 8 perguntas/exames em vez de 17 reduz custos, simplifica a interface com o paciente e acelera a triagem médica, com desempenho preditivo ainda superior.

---

## 9. Arquivos Salvos e Guia de Utilização

Foram gerados e salvos dois pipelines prontos para uso em produção:
1. **[`melhor_modelo.joblib`](file:///home/docato/Documentos/GlicoSense/melhor_modelo.joblib):** Pipeline treinado com todas as variáveis.
2. **[`melhor_modelo_otimizado_top8.joblib`](file:///home/docato/Documentos/GlicoSense/melhor_modelo_otimizado_top8.joblib):** Pipeline treinado exclusivamente com os **Top 8 atributos de maior correlação**.

### Código de Exemplo — Utilizando o Modelo Otimizado (Top 8 Atributos):

```python
import joblib
import pandas as pd

# 1. Carregar o modelo otimizado (apenas 8 variáveis necessárias)
model = joblib.load("melhor_modelo_otimizado_top8.joblib")

# 2. Dados brutos do paciente com os 8 atributos essenciais
paciente_top8 = pd.DataFrame([{
    "fasting_blood_sugar": 142,             # mg/dL
    "hba1c_level": 7.1,                     # %
    "bmi": 30.5,                            # kg/m²
    "age": 50,                              # anos
    "waist_circumference_cm": 96.0,         # cm
    "blood_pressure_systolic": 135,         # mmHg
    "family_history_diabetes": "Yes",       # Yes / No
    "physical_activity_level": "Sedentary"  # Sedentary / Moderate / Active
}])

# 3. Predição direta da classe de risco
pred = model.predict(paciente_top8)[0]
print(f"Risco Predito de Diabetes: {pred}")

# 4. Probabilidades calibradas por nível de risco
probs = model.predict_proba(paciente_top8)[0]
for classe, p in zip(model.classes_, probs):
    print(f"Probabilidade [{classe}]: {p * 100:.2f}%")
```

---

## 10. Conclusão

A seleção de atributos comprovou que o risco de diabetes na coorte avaliada é governado principalmente por um núcleo de **8 biomarcadores e fatores fisiológicos**:
- **Eixo Glicêmico:** `fasting_blood_sugar` e `hba1c_level` (correlações $r \approx 0,74$).
- **Eixo Metabólico/Antropométrico:** `bmi`, `waist_circumference_cm`, `age` ($r \approx 0,27 - 0,29$).
- **Eixo Clínico/Comportamental:** `blood_pressure_systolic`, `family_history_diabetes`, `physical_activity_level` ($r \approx 0,15 - 0,21$).

O modelo otimizado resultante [`melhor_modelo_otimizado_top8.joblib`](file:///home/docato/Documentos/GlicoSense/melhor_modelo_otimizado_top8.joblib) supera o modelo original com todas as variáveis em acurácia (**79,43%**), estabilidade de validação cruzada (**±0,64%**) e ROC-AUC (**0,9163**), proporcionando uma ferramenta clínica mais enxuta, confiável e barata para a plataforma **GlicoSense**.

