import joblib
import pandas as pd

# Carregar o modelo enxuto com as 8 variáveis de maior correlação
model = joblib.load("melhor_modelo.joblib")

# Dados do paciente apenas com os 8 atributos essenciais
paciente = pd.DataFrame([{
    "fasting_blood_sugar": 145,             # Glicemia de jejum (mg/dL)
    "hba1c_level": 7.2,                     # Hemoglobina glicada (%)
    "bmi": 31.0,                            # IMC
    "age": 36,                              # Idade
    "waist_circumference_cm": 98.0,         # Circunferência da cintura (cm)
    "blood_pressure_systolic": 138,         # Pressão sistólica (mmHg)
    "family_history_diabetes": "No",       # Histórico familiar (Yes/No)
    "physical_activity_level": "Sedentary"  # Atividade (Sedentary/Moderate/Active)
}])

# Predição de risco e probabilidades
pred = model.predict(paciente)[0]
probs = model.predict_proba(paciente)[0]

print(f"Risco Estimado: {pred}")
for classe, prob in zip(model.classes_, probs):
    print(f"  - {classe}: {prob * 100:.1f}%")