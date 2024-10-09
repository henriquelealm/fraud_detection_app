# app.py

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os

# Importar a função de geração de transações do transaction_generator.py
from transaction_generator import generate_transactions

app = Flask(__name__)

# Definir a lista de features na ordem correta
FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Carregar o modelo treinado e o scaler
# Certifique-se de que 'fraud_model.pkl' e 'scaler.pkl' estão na mesma pasta que 'app.py'
with open('fraud_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Lista para armazenar transações suspeitas
alerts = []

# Lista para armazenar todas as transações geradas
transactions = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            quantity = int(request.form.get('quantity'))
            if quantity <= 0:
                raise ValueError("A quantidade deve ser um número positivo.")
            
            # Definir a quantidade de fraudes e legítimas (30% fraudes)
            n_fraud = int(quantity * 0.3)
            n_legit = quantity - n_fraud
            
            # Gerar transações usando o gerador fornecido
            synthetic_df = generate_transactions(n_fraud, n_legit)
            
            if synthetic_df.empty:
                raise ValueError("Falha ao gerar transações sintéticas.")
            
            # Selecionar apenas as colunas necessárias para a previsão
            X = synthetic_df[FEATURE_NAMES]
            
            # Fazer previsões
            predictions = model.predict(X)
            synthetic_df['Model_Prediction'] = predictions
            
            # Adicionar transações ao histórico
            transactions.extend(synthetic_df.to_dict(orient='records'))
            
            # Filtrar transações fraudulentas
            fraudulent_transactions = synthetic_df[synthetic_df['Model_Prediction'] == 1][['Time', 'Amount']].to_dict(orient='records')
            alerts.extend(fraudulent_transactions)
            
            # Converter para HTML apenas as transações fraudulentas
            results = fraudulent_transactions
            
            return render_template('results.html', results=results, quantity=quantity)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

@app.route('/alerts')
def view_alerts():
    return render_template('alerts.html', alerts=alerts)

@app.route('/dashboard')
def dashboard():
    # Contagem total de transações e fraudes
    total_transactions = len(transactions)
    fraud_transactions = sum(1 for t in transactions if t['Model_Prediction'] == 1)
    
    # Preparar dados para o gráfico de Amount e Time
    time_values = [t['Time'] for t in transactions]
    amount_values = [t['Amount'] for t in transactions]
    
    return render_template(
        'dashboard.html', 
        total_transactions=total_transactions,
        fraud_transactions=fraud_transactions,
        time_values=time_values,
        amount_values=amount_values
    )

if __name__ == '__main__':
    app.run(debug=True)
