# transaction_generator.py

import pandas as pd
import numpy as np
import pickle
import os

# Definir a lista de features na ordem correta
FEATURE_NAMES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

def load_model_and_scaler(model_path='fraud_model.pkl', scaler_path='scaler.pkl'):
    """Carrega o modelo treinado e o scaler a partir dos arquivos pickle."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        print("Modelo e scaler carregados com sucesso.")
        return model, scaler
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Certifique-se de que 'fraud_model.pkl' e 'scaler.pkl' estão no diretório atual.")
        raise
    except Exception as e:
        print(f"Erro ao carregar os arquivos: {e}")
        raise

def load_fraud_examples(csv_path='fraud_examples.csv'):
    """Carrega os exemplos de transações fraudulentas a partir de um arquivo CSV."""
    if not os.path.exists(csv_path):
        print(f"Erro: O arquivo '{csv_path}' não foi encontrado.")
        raise FileNotFoundError(f"Arquivo '{csv_path}' não encontrado.")
    try:
        fraud_df = pd.read_csv(csv_path)
        # Remover a coluna 'index' se presente
        if 'index' in fraud_df.columns:
            fraud_df = fraud_df.drop(columns=['index'])
        # Garantir que todas as features necessárias estão presentes
        missing_features = set(FEATURE_NAMES) - set(fraud_df.columns)
        if missing_features:
            print(f"Erro: As seguintes features estão faltando nos exemplos de fraude: {missing_features}")
            raise ValueError(f"Features faltantes: {missing_features}")
        # Selecionar apenas as features necessárias e na ordem correta
        fraud_df = fraud_df[FEATURE_NAMES]
        print(f"{len(fraud_df)} exemplos de fraude carregados.")
        return fraud_df
    except Exception as e:
        print(f"Erro ao carregar os exemplos de fraude: {e}")
        raise

def calculate_feature_stats(fraud_df):
    """Calcula a média e o desvio padrão de cada feature a partir dos exemplos de fraude."""
    feature_means = fraud_df.mean()
    feature_stds = fraud_df.std()
    return feature_means, feature_stds

def generate_synthetic_fraud(feature_means, feature_stds, n_fraud):
    """Gera transações sintéticas fraudulentas baseadas nas estatísticas calculadas."""
    synthetic_fraud = {}
    for feature in FEATURE_NAMES:
        mean = feature_means[feature]
        std = feature_stds[feature]
        # Gerar valores normalmente distribuídos
        synthetic_fraud[feature] = np.random.normal(loc=mean, scale=std, size=n_fraud)
    fraud_df = pd.DataFrame(synthetic_fraud)
    # Garantir que os valores de 'Time' e 'Amount' sejam positivos
    fraud_df['Time'] = fraud_df['Time'].apply(lambda x: max(x, 0))
    fraud_df['Amount'] = fraud_df['Amount'].apply(lambda x: max(x, 0))
    fraud_df['Class'] = 1  # Label real de fraude
    return fraud_df

def generate_synthetic_legit(n_legit, feature_means, feature_stds):
    """
    Gera transações sintéticas legítimas.
    Para simplificar, assume-se que as transações legítimas têm diferentes distribuições.
    Você pode ajustar as médias e desvios padrão conforme necessário.
    """
    synthetic_legit = {}
    for feature in FEATURE_NAMES:
        if feature == 'Time':
            mean = feature_means[feature] + np.abs(feature_means[feature]) * 0.5  # Maior que a média de fraude
            std = feature_stds[feature] * 1.2
        elif feature == 'Amount':
            mean = feature_means[feature] - np.abs(feature_means[feature]) * 0.5  # Menor que a média de fraude
            std = feature_stds[feature] * 1.2
        else:
            mean = feature_means[feature] * 0.8  # Ajuste conforme necessário
            std = feature_stds[feature] * 1.2
        synthetic_legit[feature] = np.random.normal(loc=mean, scale=std, size=n_legit)
    legit_df = pd.DataFrame(synthetic_legit)
    # Garantir que os valores de 'Time' e 'Amount' sejam positivos
    legit_df['Time'] = legit_df['Time'].apply(lambda x: max(x, 0))
    legit_df['Amount'] = legit_df['Amount'].apply(lambda x: max(x, 0))
    legit_df['Class'] = 0  # Label real de transação legítima
    return legit_df

def generate_transactions(n_fraud, n_legit, csv_path='fraud_examples.csv'):
    """
    Gera um conjunto de transações sintéticas com uma porcentagem específica de fraudes.
    
    Args:
        n_fraud (int): Número de transações fraudulentas a serem geradas.
        n_legit (int): Número de transações legítimas a serem geradas.
        csv_path (str): Caminho para o arquivo CSV contendo exemplos de fraudes.
        
    Returns:
        pd.DataFrame: DataFrame contendo as transações geradas com colunas originais e 'Model_Prediction'.
    """
    try:
        model, scaler = load_model_and_scaler()
        fraud_examples_df = load_fraud_examples(csv_path)
    except Exception as e:
        print(f"Erro durante o carregamento do modelo ou dos exemplos de fraude: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio em caso de erro
    
    feature_means, feature_stds = calculate_feature_stats(fraud_examples_df)
    
    print(f"Gerando {n_fraud} transações fraudulentas e {n_legit} transações legítimas.")
    
    # Gerar transações sintéticas fraudulentas
    synthetic_fraud_df = generate_synthetic_fraud(feature_means, feature_stds, n_fraud)
    
    # Gerar transações sintéticas legítimas
    synthetic_legit_df = generate_synthetic_legit(n_legit, feature_means, feature_stds)
    
    # Combinar as transações
    synthetic_df = pd.concat([synthetic_fraud_df, synthetic_legit_df], ignore_index=True)
    
    # Aplicar o pré-processamento (escalonamento)
    # Supondo que apenas 'Amount' foi escalonado durante o treinamento
    synthetic_df['Amount'] = scaler.transform(synthetic_df[['Amount']])
    
    # Reorganizar as colunas para corresponder ao modelo
    # Excluir a coluna 'Class' para previsão
    X = synthetic_df[FEATURE_NAMES]
    
    # Realizar previsões
    predictions = model.predict(X)
    synthetic_df['Model_Prediction'] = predictions
    
    # Adicionar a coluna 'Class' original
    synthetic_df['Class'] = synthetic_df['Class']
    
    return synthetic_df
