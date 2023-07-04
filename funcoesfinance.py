import yfinance as yf

import pandas_datareader as pdr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn import linear_model, datasets
import numpy as np

#Funiconando

def dados(simbolo,dataini,datafim,tocsv): #Funciona
	# aapl = pdr.get_data_yahoo('ABEV3.SA', 
                          # start=datetime.datetime(2010, 1, 1), 
                          # end=datetime.datetime(2020, 1, 1))
	if tocsv == True:
		dataSimbolo = pdr.get_data_yahoo(simbolo, 
                          start=dataini, 
                          end=datafim)
		dataSimbolo.to_csv(simbolo + '.csv')
	else:
		dataSimbolo = pd.read_csv(simbolo + '.csv')

	return(dataSimbolo)

def stockinfo(simbolo):
	stock = yf.Ticker(simbolo)
	informacoes = stock.info
	return(informacoes)

def donchian(dataSimbolo,janela): #Funciona

	high_px = dataSimbolo['High']
	low_px = dataSimbolo['Low']

	dataSimbolo['doch+'] = high_px.rolling(window=janela).max()
	dataSimbolo['doch-'] = low_px.rolling(window=janela).min()

	return(dataSimbolo)

def MA(dataSimbolo,janela): #Funciona
	close_px = dataSimbolo['Close']
	dataSimbolo['MA'+str(janela)] = close_px.rolling(window=janela).mean()

	return(dataSimbolo)

def Momentum(dataSimbolo,janela):#Funciona
	#janela = 101 aprox 6 meses
	dataSimbolo['mom'+str(janela)] = dataSimbolo['Close'].pct_change(janela)

	return(dataSimbolo)

def MomentumDeriv(dataSimbolo,janela):#Funciona
	#janela = 101 aprox 6 meses
	# print(dataSimbolo.tail())
	# dataSimbolo['momder'] = dataSimbolo['mom'+str(janela)].iloc[-2]-dataSimbolo['mom'+str(janela)].iloc[-1]
	dataSimbolo['momder'] = dataSimbolo['mom'+str(janela)].diff(1)

	return(dataSimbolo)

def volumeIndice(dataSimbolo,janela):
	volume = dataSimbolo['Volume']
	# print(volume)
	dataSimbolo['Vol'+str(janela)] = volume.rolling(window=janela).mean()
	# print(dataSimbolo.tail(10))
	volIndice = dataSimbolo['Volume'].iloc[-1]/dataSimbolo['Vol'+str(janela)].iloc[-1]
	# print(volIndice)
	return volIndice

# Funcoes LongShort ######################################

#FUNCIONANDO ########## Calcular a regressão linear
def linearRegressao(valores):
	x = np.linspace(1, len(valores), len(valores))
	# print(x)
	a, b = np.polyfit(x, valores, deg=1)
	# print('a = ' + str(a))
	# print('b = ' + str(b))
	y_est = a * x + b
	return [y_est,a,b]
	# return y_est

def meiaVida(valores):
	#valores deve ser um residuo
	lag = np.roll(valores,1)
	lag[0] = 0
	ret = valores - lag

	lag2 = sm.add_constant(lag)

	model = sm.OLS(ret,lag2)
	res = model.fit()

	MV = round(-np.log(2)/res.params[1])

	return MV

#Funcionando ########## Calcular a regressão RANSAC eliminando outliers
def RANSAC(Ativo1,Ativo2):
	x = Ativo1.iloc[:].values
	y = Ativo2.iloc[:].values

	# print(type(x))
	X = x.reshape(-1,1)

	# Fit line using all data
	lr = linear_model.LinearRegression()
	lr.fit(X, y)

	# Robustly fit linear model with RANSAC algorithm
	ransac = linear_model.RANSACRegressor()
	ransac.fit(X, y)
	inlier_mask = ransac.inlier_mask_
	outlier_mask = np.logical_not(inlier_mask)

	# Predict data of estimated models
	line_X = np.arange(X.min(), X.max())[:, np.newaxis]
	# print(line_X)
	line_y = lr.predict(line_X)
	# print(line_y)
	line_y_ransac = ransac.predict(line_X)
	# print(line_y_ransac)
	# print(x)
	# print(len(x[inlier_mask]))
	# print(len(y[inlier_mask]))
	Rvalor = x[inlier_mask]/y[inlier_mask]
	y_estimado = linearRegressao(Rvalor)
	Rresiduo = residuo(Rvalor,y_estimado[0])

	#################### Score e sum() geram os numeros para verificação da quantidade de outliers removidos
	# print('Score: ' + str(ransac.score(X, y)))
	# print('inlier_mask')
	# print(inlier_mask.sum())
	# print('outlier_mask')
	# print(outlier_mask.sum())

	#####################Como gerar graficos com RANSAC e linhas de regrassão
	# pyplot.scatter(
	#     X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	# )
	# pyplot.scatter(
	#     X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
	# )
	# # pyplot.scatter(Ativo1,Ativo2)
	# pyplot.plot(line_X, line_y, color="navy", label="Linear regressor")
	# pyplot.plot(line_X,line_y_ransac,
	#     color="cornflowerblue",
	#     label="RANSAC regressor",
	# )

	# pyplot.show()

	#####################Pode-se usar o resultado da remoção dos outliers para testes estatisticos Spearman, Pearson ou Dickey Fulle
	# ativo1 = x[inlier_mask]
	# ativo2 = y[inlier_mask]

	return [Rresiduo,ransac.score(X, y)]

#Funcionando ########## Calcular o residuo
def residuo(valor,y_est):
	#y_est é resultado do retorno de linearRegresao() ou RANSAC()
	residuo = valor - y_est
	residuostd = residuo/residuo.std()

	#residuostd será usado na tomada de decisão estatistica
	return [residuo,residuostd]

#FUNCIONANDO ########## Calcular a correlacao Spearman e Pearsons
def correlacaoPS(Ativo1,Ativo2):
	# Ativo1 e Ativo2 devem ter o mesmo tamanho e ser obtido apartir de
	#PRIO3 = tabela['PRIO3'].iloc[-intervalo:]
	#onde intervalo é o tamanho da amostra sendo testada

	corrpearson,_ = stats.pearsonr(Ativo1,Ativo2)
	# print('Pearsons correlation %.f: %.3f' % (intervalo,corr))

	corrspearman,_ = stats.spearmanr(Ativo1,Ativo2)
	# print('Spearmans correlation: %.f: %.3f' % (intervalo,corr))

	return [corrpearson,corrspearman]


#Work ########## Calcular o angulo de rotacao
#Work ########## Calcular o Beta Rotation

#Funcionando ########## Calcular o Dikey Fuller
def ADF(residuo):
	# O ADF deve ser calcula a partir de uma serie de residuos de regressão linear ou RANSAC

	result = adfuller(residuo, autolag='AIC')

	######como obter os dados de retorno

	# print(f'ADF Statistic: {result[0]}')
	# print(f'n_lags: {result[1]}')
	# print(f'p-value: {result[1]}')
	# for key, value in result[4].items():
	#     print('Critial Values:')
	#     print(f'   {key}, {value}')  
	return result

def LongShortTable(Ativo1, Ativo2,tamanho):
	# tamanho = [90,110,130,150,170,190,210]
	# tamanho = [190]

	tabelaLongShort = pd.DataFrame(index=tamanho, columns=['Residuo','ADF','ADF%','Pearson','Rscore','RResiduo','RADF','Std','Angulo','Meia Vida'])

	# Ativo1 = tabela['SIMH3']
	# Ativo2 = tabela['MOVI3']

	#Intervalo - ADF - ADF % - Pearson - Spearmann - RANSAC Score - RANSAC ADF - Angulo - Meia Vida

	for intervalo in tamanho:
		Ativo1strip = Ativo1.iloc[-intervalo:] #Estava invertido o y com o x, por isso troca 1 por 2
		Ativo2strip = Ativo2.iloc[-intervalo:]

		#Calcular o Residuo
		# print('############################################')
		# print('intervalo = ' + str(intervalo))
		valores = Ativo1strip/Ativo2strip
		# print(valores)
		linear = linearRegressao(valores)
		# print(linear[0])
		# print(y_est)
		residuo1 = residuo(valores,linear[0])
		tabelaLongShort.loc[intervalo,'Std'] = residuo1[0].std().round(3)

		a, b = np.polyfit(Ativo2strip, Ativo1strip, deg=1)
		tabelaLongShort.loc[intervalo,'Angulo'] = a.round(2)


		#Calcular a meia vida
		# print(residuo1[1])
		MV = meiaVida(residuo1[1])
		tabelaLongShort.loc[intervalo,'Meia Vida'] = MV

		#Calculo do ADF
		tabelaLongShort.loc[intervalo,'Residuo'] = residuo1[1].iloc[-1].round(2)
		ADF1 = ADF(residuo1[1])
		# print(f'ADF1 Statistic: {ADF1[0]}')
		# print(f'n_lags: {ADF1[1]}')
		# print(f'p-value: {ADF1[1]}')
		ADFperc = '0%'
		ADFcert = True
		for key, value in ADF1[4].items():
		    # print('Critial Values:')
		    # print(f'   {key}, {value}')
		    if ADF1[0]<value and ADFcert:
		    	ADFperc = key
		    	ADFcert = False

		tabelaLongShort.loc[intervalo,'ADF'] = ADF1[0].round(2)
		tabelaLongShort.loc[intervalo,'ADF%'] = ADFperc
		
		corr = correlacaoPS(Ativo1strip,Ativo2strip)
		# print(corr)

		tabelaLongShort.loc[intervalo,'Pearson'] = corr[0].round(2)

		ransac1 = RANSAC(Ativo1strip,Ativo2strip)
		Rresiduo = ransac1[0][1]
		# print(ransac)
		# print(Rresiduo[-1])
		# print(ransac[1])

		tabelaLongShort.loc[intervalo,'Rscore'] = ransac1[1].round(2)
		tabelaLongShort.loc[intervalo,'RResiduo'] = Rresiduo[-1].round(2)

		# print('Rresiduo = ')
		# print(Rresiduo)
		RADF = ADF(Rresiduo)
		tabelaLongShort.loc[intervalo,'RADF'] = RADF[0].round(2)

	# print(tabelaLongShort)


	return tabelaLongShort
