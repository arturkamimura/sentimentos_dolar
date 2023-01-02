import pandas as pd

df = pd.read_excel('teste_folha2.xlsx')

df_dolar = pd.read_csv('dados_dolar.csv', decimal=',')


df2 = df.merge(df_dolar)

df2 = df2.drop(['Data','Último', 'Abertura', 'Máxima', 'Mínima', 'Vol.'],  axis=1)

#df2['Var'].replace(',','.')
df2['Var'] = pd.to_numeric(df2['Var'])

df2.to_excel('compilado_folha.xlsx', index=None)
