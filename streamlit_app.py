from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import streamlit as st
import joblib
import sklearn
import numpy as np
import pandas as pd
import os


def criarModelo(modelo, base01, alvo):
    x_train, x_test, y_train, y_test = train_test_split(base01.drop(alvo, axis=1),
                                                        base01[alvo],
                                                        test_size=0.3,
                                                        random_state=42)

    modeloRegressor = modelo
    modeloRegressor.fit(x_train, y_train)
    return modeloRegressor


def importarModelo(modeloEscolhido):
    url = 'streamlit/predicao_sem_historico_das_empresas/base01/dataset_sem_coluna_empresa.csv'
    alvo = 'IMPOSTOS_ESTADUAIS'
    base01 = pd.read_csv(url, engine="openpyxl")
    
    dados = {
    'RECEITA_VENDAS_BENS_OU_SERVICOS': RECEITA_VENDAS_BENS_OU_SERVICOS,
    'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS': CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS,
    'DESPESAS_RECEITAS_OPERACIONAIS': DESPESAS_RECEITAS_OPERACIONAIS,
    'RESULTADO_FINANCEIRO': RESULTADO_FINANCEIRO,
    'RECEITAS': RECEITAS,
    'DISTRIBUICAO_DO_VALOR_ADICIONADO': DISTRIBUICAO_DO_VALOR_ADICIONADO,   
}
    
    atributosSelecionados = ['RECEITA_VENDAS_BENS_OU_SERVICOS', 'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS', 'DESPESAS_RECEITAS_OPERACIONAIS', 
                             'RESULTADO_FINANCEIRO', 'RECEITAS', 'DISTRIBUICAO_DO_VALOR_ADICIONADO', 'IMPOSTOS_ESTADUAIS']

    base01 = base01.loc[:, atributosSelecionados]

    return criarModelo(modeloEscolhido, base01, alvo)


modeloET = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               max_samples=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, min_samples_leaf=1,
                               min_samples_split=2, min_weight_fraction_leaf=0.0,
                               n_estimators=100, n_jobs=-1, oob_score=False,
                               random_state=2444, verbose=0, warm_start=False)

modeloGB = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                                     init=None, learning_rate=0.1, loss='ls', max_depth=3,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0.0, min_impurity_split=None,
                                     min_samples_leaf=1, min_samples_split=2,
                                     min_weight_fraction_leaf=0.0, n_estimators=100,
                                     n_iter_no_change=None, presort='deprecated',
                                     random_state=2444, subsample=1.0, tol=0.0001,
                                     validation_fraction=0.1, verbose=0, warm_start=False)

modeloRF = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                 max_depth=None, max_features='auto', max_leaf_nodes=None,
                                 max_samples=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, min_samples_leaf=1,
                                 min_samples_split=2, min_weight_fraction_leaf=0.0,
                                 n_estimators=100, n_jobs=-1, oob_score=False,
                                 random_state=2444, verbose=0, warm_start=False)


modelo_options = ['Extra Trees', 'Gradient Boosting', 'Random Forest']

modelo_values = {'Extra Trees': modeloET, 'Gradient Boosting': modeloGB, 'Random Forest': modeloRF}


st.header('Modelo de Predição de Empresas Sonegadoras de Impostos Estaduais - Sem Histórico das Empresas')
st.subheader('Preencha as informações solicitadas para obter a predição:')

# Definindo os campos de entrada
modelo_value = st.selectbox('Escolha o Modelo', options=modelo_options)
modeloSelecionado = modelo_values.get(modelo_value)

receita_com_vendas_value = st.number_input('Qual o valor das Receitas com Vendas de Bens ou Serviços ?')
receita_com_vendas = receita_com_vendas_value.get(receita_com_vendas_value)

custo_dos_bens_vendidos_value = st.number_input('Qual o valor do Custo dos Bens ou Serviços Vendidos ?')
custo_dos_bens_vendidos = custo_dos_bens_vendidos_value.get(custo_dos_bens_vendidos_value)

despesas_receitas_operacionais_value = st.number_input('Qual o valor das Despesas e Receitas Operacionais ?')
despesas_receitas_operacionais = despesas_receitas_operacionais_value.get(despesas_receitas_operacionais_value)

resultado_financeiro_value = st.number_input('Qual o valor do Resultado Financeiro ?')
esultado_financeiro = resultado_financeiro_value.get(resultado_financeiro_value)

receitas_value = st.number_input('Qual o valor das Receitas ?')
receitas = receitas_value.get(receitas_value)

distribuicao_do_valor_adicionado_value = st.number_input('Qual o valor da Distribuição do valor Adicionado ?')
distribuicao_do_valor_adicionado = distribuicao_do_valor_adicionado_value.get(distribuicao_do_valor_adicionado_value)


dados = {
    'RECEITA_VENDAS_BENS_OU_SERVICOS': RECEITA_VENDAS_BENS_OU_SERVICOS,
    'CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS': CUSTO_DOS_BENS_OU_SERVICOS_VENDIDOS,
    'DESPESAS_RECEITAS_OPERACIONAIS': DESPESAS_RECEITAS_OPERACIONAIS,
    'RESULTADO_FINANCEIRO': RESULTADO_FINANCEIRO,
    'RECEITAS': RECEITAS,
    'DISTRIBUICAO_DO_VALOR_ADICIONADO': DISTRIBUICAO_DO_VALOR_ADICIONADO,   
}

modelo = importarModelo(modeloSelecionado)
botao = st.button('Efetuar Predição')
if(botao):  
    dadosFormatados = pd.DataFrame([dados])
    resultado = modelo.predict_proba(dadosFormatados)
    prob =  round(resultado[0][0] * 100, 3)
    st.write('Probabilidade de Sonegação: ', prob, ' %')
  
