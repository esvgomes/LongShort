import funcoesfinance as ff
from matplotlib import pyplot
import numpy as np
import datetime
import pandas as pd


simbolos = ['PRIO3','LCAM3','TASA4','SIMH3','CARD3','BOVA11','BPAC11','SHUL4','MOVI3']

hoje = datetime.datetime.now()
inicio = hoje - datetime.timedelta(days = 365)
ontem = hoje - datetime.timedelta(days = 1)

for sim in simbolos:
	# infos = ff.stockinfo(sim + '.SA')
	# print(infos)
	print(sim)
	dataSimbol = ff.dados(sim + '.SA',
							inicio,
							ontem,
							True)

tabela = pd.DataFrame()

for sim in simbolos:
	dados = pd.read_csv(sim + '.SA.csv')
	tabela[sim]=dados['Close']

# print(tabela.tail(10))

print('dados OK')

tamanho = [90,110,130,150,170,190,210]
# tamanho = [190]

# tabelaLongShort = pd.DataFrame(index=tamanho, columns=['Residuo','ADF','ADF%','Pearson','Rscore','RResiduo','RADF','Angulo','Meia Vida'])

correlacao = tabela.corr()
# print(correlacao)

#WORK ########## Fazer a tabela LongShort 2 ativos

pairs = list()

for sim in simbolos:
	for sim2 in simbolos:

		if sim == sim2: continue
		print(sim + '/' + sim2)

		Ativo1 = tabela[sim]
		Ativo2 = tabela[sim2]

		valores = Ativo1/Ativo2
		tabelaLS = ff.LongShortTable(Ativo1, Ativo2,tamanho)
		
		print(tabelaLS)

		# if tabela['Residuo'].values >1.5:
		# 	print('ESPECIAL ' + sim + '/' + sim2)

		if ((tabelaLS['Residuo'] > 1.8).any() or (tabelaLS['Residuo'] < -1.8).any()) and (tabelaLS['ADF'] < -3).any():
			pairs.append(sim + '/' + sim2)
			
print(pairs)

# import statsmodels.api as sm

# y_est = ff.linearRegressao(valores)
# residuo1 = ff.residuo(valores,y_est)

# z_lag = np.roll(residuo1[1],1)
# z_lag[0] = 0
# # print(residuo1[1])
# # print('z_lag')
# # print(z_lag)
# z_ret = residuo1[1] - z_lag
# # z_ret[0] = 0
# # print('z_ret')
# # print(z_ret)
# z_lag2 = sm.add_constant(z_lag)

# model = sm.OLS(z_ret,z_lag2)
# res = model.fit()

# print(res)

# HL = round(-np.log(2)/res.params[1])

# print('Meia Vida =' + str(HL))


# y_est = ff.linearRegressao(valores)
# residuo1 = ff.residuo(valores,y_est)

# y_linear = ff.linearRegressao(residuo1[1])
# # print(valores)
# # print(y_est)
# # print (y_linear)
# #Calcular a meia vida
# MV = ff.meiaVida(residuo1[1])

#Intervalo - ADF - ADF % - Pearson - Spearmann - RANSAC Score - RANSAC ADF - Angulo - Meia Vida

# for intervalo in tamanho:
# 	Ativo1strip = Ativo1.iloc[-intervalo:]
# 	Ativo2strip = Ativo2.iloc[-intervalo:]

# 	#Calcular o Residuo
# 	valores = Ativo1strip/Ativo2strip
# 	y_est = ff.linearRegressao(valores)
# 	residuo = ff.residuo(valores,y_est)

# 	tabelaLongShort.loc[intervalo,'Residuo'] = residuo[1].iloc[-1]
# 	ADF = ff.ADF(residuo[1])
# 	# print(f'ADF Statistic: {ADF[0]}')
# 	# print(f'n_lags: {ADF[1]}')
# 	# print(f'p-value: {ADF[1]}')
# 	ADFperc = '0%'
# 	ADFcert = True
# 	for key, value in ADF[4].items():
# 	    # print('Critial Values:')
# 	    # print(f'   {key}, {value}')
# 	    if ADF[0]<value and ADFcert:
# 	    	ADFperc = key
# 	    	ADFcert = False

# 	tabelaLongShort.loc[intervalo,'ADF'] = ADF[0]
# 	tabelaLongShort.loc[intervalo,'ADF%'] = ADFperc
	
# 	corr = ff.correlacaoPS(Ativo1strip,Ativo2strip)
# 	# print(corr)

# 	tabelaLongShort.loc[intervalo,'Pearson'] = corr[0]

# 	ransac = ff.RANSAC(Ativo1strip,Ativo2strip)
# 	Rresiduo = ransac[0][1]
# 	# print(ransac)
# 	# print(Rresiduo[-1])
# 	# print(ransac[1])

# 	tabelaLongShort.loc[intervalo,'Rscore'] = ransac[1]
# 	tabelaLongShort.loc[intervalo,'RResiduo'] = Rresiduo[-1]

# 	RADF = ff.ADF(Rresiduo)
# 	tabelaLongShort.loc[intervalo,'RADF'] = RADF[0]

# print(tabelaLongShort)


	# x = np.linspace(1, len(Rresiduo), len(Rresiduo))
	# fig, ax = pyplot.subplots()
	# pyplot.title('interlavo %i'%(intervalo))

	# # ax[0].plot(x,residuo[0], marker='.')

	# ax.plot(x, np.zeros(len(Rresiduo)), '-', alpha=0.4, color='b')
	# ax.plot(x, np.zeros(len(Rresiduo))+2, '-',alpha=0.6,color='r')
	# ax.plot(x, np.zeros(len(Rresiduo))-2, '-',alpha=0.6,color='r')
	# ax.plot(x,Rresiduo, marker='.')
	# for xy in zip(x, Rresiduo):
	#     if xy[1] > 2 or xy[1] < -2 or xy[0] == len(Rresiduo):
	#     	ax.annotate('%s' % str(xy[1])[:4], xy=xy, textcoords='data')

	# pyplot.show()
##########################################################################

#WORK ########## Fazer a curva Stock/Stock

# intervalo = 90

# PRIO3 = tabela['PRIO3'].iloc[-intervalo:]
# LCAM3 = tabela['LCAM3'].iloc[-intervalo:]
# TASA4 = tabela['TASA4'].iloc[-intervalo:]
# SIMH3 = tabela['SIMH3'].iloc[-intervalo:]
# CARD3 = tabela['CARD3'].iloc[-intervalo:]
# BOVA11 = tabela['BOVA11'].iloc[-intervalo:]
# BPAC11 = tabela['BPAC11'].iloc[-intervalo:]
# SHUL4 = tabela['SHUL4'].iloc[-intervalo:]
# MOVI3 = tabela['MOVI3'].iloc[-intervalo:]
# XPBR31 = tabela['XPBR31'].iloc[-intervalo:]

# print(MOVI3.tail())
# print(LCAM3.tail())
# Ativo1 = BPAC11
# Ativo2 = BOVA11

# At1STR = 'BPAC11'
# At2STR = 'BOVA11'

# valor = Ativo1/Ativo2
# print(valor.tail())



# Calculo do residuo



######


#FUNCIONANDO ########## Calcular a regressão linear
# x = np.linspace(1, len(valor), len(valor))
# a, b = np.polyfit(x, valor, deg=1)
# y_est = a * x + b

# #Funcionando ########## Calcular o residuo
# residuo = valor - y_est
# residuostd = residuo/residuo.std()

# #FUNCIONANDO ########## Calcular a correlacao Spearman e Pearsons
# from scipy import stats

# corr,_ = stats.pearsonr(Ativo1,Ativo2)
# print('Pearsons correlation %.f: %.3f' % (intervalo,corr))

# corr,_ = stats.spearmanr(Ativo1,Ativo2)
# print('Spearmans correlation: %.f: %.3f' % (intervalo,corr))

# #Work ########## Calcular o angulo de rotacao
# #Work ########## Calcular o Beta Rotation
# #Funcionando ########## Calcular o Dikey Fuller

# from statsmodels.tsa.stattools import adfuller

# result = adfuller(residuo, autolag='AIC')

# print(f'ADF Statistic: {result[0]}')
# print(f'n_lags: {result[1]}')
# print(f'p-value: {result[1]}')
# for key, value in result[4].items():
#     print('Critial Values:')
#     print(f'   {key}, {value}')    

# #WORK ########## Calcular o Fisher MaxMin




# #Funcionando ####### Plotar um par especifico  ##############

# #Print com valores de desvios de residuos
# fig, ax = pyplot.subplots()
# pyplot.title('%s/%s  interlavo %i'%(At1STR,At2STR,intervalo))

# ax.plot(x, np.zeros(intervalo), '-', alpha=0.4, color='b')
# ax.plot(x, np.zeros(intervalo)+2, '-',alpha=0.6,color='r')
# ax.plot(x, np.zeros(intervalo)-2, '-',alpha=0.6,color='r')
# ax.plot(x,residuostd, marker='.',picker=True)
# for xy in zip(x, residuostd):                                       # <--
#     if xy[1] > 2 or xy[1] < -2 or xy[0] == intervalo:
#     	ax.annotate('%s' % str(xy[1])[:4], xy=xy, textcoords='data')

# #     ind = event.ind
# #     print('onpick3 scatter:', ind, np.take(x, ind), np.take(residuostd, ind))

# # fig.canvas.mpl_connect('pick_event', onpick3)

# pyplot.show()

# #Print com valores absolutos de residuos
# # fig, ax = pyplot.subplots()
# # ax.plot(x, np.zeros(30), '-', alpha=0.4, color='b')
# # ax.plot(x, np.zeros(30)+(2*residuo.std()), '-',alpha=0.6)
# # ax.plot(x, np.zeros(30)+(-2*residuo.std()), '-',alpha=0.6)
# # ax.plot(x,residuo)
# # pyplot.show()

# # FUNCIONANDO ####### Plotar o heat map  ##############
# # fig, ax = pyplot.subplots()
# # im = ax.imshow(correlacao)

# # ax.set_xticks(np.arange(len(simbolos)))
# # ax.set_xticklabels(simbolos)
# # ax.set_yticks(np.arange(len(simbolos)))
# # ax.set_yticklabels(simbolos)

# # pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right",
# #          rotation_mode="anchor")

# # for i in range(len(simbolos)):
# #     for j in range(len(simbolos)):
# #         valor = str(correlacao.iloc[i, j])
# #         if len(valor) > 5: valor = valor[:5]
# #         print(valor)
# #         text = ax.text(j, i, valor[0:4],ha="center", va="center", color="w")

# # ax.set_title("Correlacao 365 dias")
# # fig.tight_layout()
# # pyplot.show()

# #WORK ########## RANSAC otimizacao

# from sklearn import linear_model, datasets

# # print(Ativo1.tail())
# # print(Ativo2.tail())

# X = Ativo1.to_numpy().reshape(-1, 1)
# y = Ativo2.to_numpy().reshape(-1, 1)

# # x = Ativo1.iloc[:].values
# # y = Ativo2.iloc[:].values

# # print(type(x))
# # X = x.reshape(-1,1)

# # Fit line using all data
# lr = linear_model.LinearRegression()
# lr.fit(X, y)

# # Robustly fit linear model with RANSAC algorithm
# ransac = linear_model.RANSACRegressor()
# ransac.fit(X, y)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)

# # Predict data of estimated models
# line_X = np.arange(X.min(), X.max())[:, np.newaxis]
# # print(line_X)
# line_y = lr.predict(line_X)
# # print(line_y)
# line_y_ransac = ransac.predict(line_X)
# print('Score: ' + str(ransac.score(X, y)))
# print('inlier_mask')
# print(inlier_mask.sum())
# print('outlier_mask')
# print(outlier_mask.sum())

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

# ##Testes estatisticos RANSAC


# residuo = (X[inlier_mask]/y[inlier_mask]) - (line_X[inlier_mask]/line_y_ransac[inlier_mask]) #(ainda não funciona)
# result = adfuller(valor, autolag='AIC')

# print('ADF RANSC Outliers')
# print(f'ADF Statistic: {result[0]}')
# # print(f'n_lags: {result[1]}')
# # print(f'p-value: {result[1]}')
# # for key, value in result[4].items():
# #     print('Critial Values:')
# #     print(f'   {key}, {value}')  

# x = Ativo1[inlier_mask]
# y = Ativo2[inlier_mask]
# print('lenx = ' + str(len(x)) + ' leny = ' + str(len(y)))
# corr,_ = stats.pearsonr(x,y)
# print('Pearsons correlation RANSAC %.f: %.3f' % (intervalo,corr))

# corr,_ = stats.spearmanr(x,y)
# print('Spearmans correlation RANSAC %.f: %.3f' % (intervalo,corr))

