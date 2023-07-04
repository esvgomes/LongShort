import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import pandas as pd
from dash.dependencies import Input, Output
import funcoesfinance as ff
import datetime
from matplotlib import pyplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

#http://localhost:8050

############# Variaveis Globais   ############################

# Simbolos das ações do IBRXX em Nov 2021
# simbolos = ['ALPA4','ABEV3','AMER3','ASAI3','AZUL4','B3SA3','BIDI4','BIDI11','BPAN4',
# 			'BBSE3','BRML3','BBDC3','BBDC4','BRAP4','BBAS3','BRKM5','BRFS3','BPAC11',
# 			'CRFB3','CCRO3','CMIG4','CESP6','CIEL3','COGN3','CPLE6','CSAN3','CPFE3',
# 			'CVCB3','CYRE3','DXCO3','ECOR3','ELET3','ELET6','EMBR3','ENBR3',
# 			'ENEV3','EGIE3','EQTL3','EZTC3','FLRY3','GGBR4','GOAU4','GOLL4',
# 			'NTCO3','SOMA3','HAPV3','HYPE3','GNDI3','IRBR3','ITSA4','ITUB4',
# 			'JBSS3','JHSF3','KLBN11','LIGT3','RENT3','LCAM3','LWSA3','LAME3','LAME4',
# 			'LREN3','MGLU3','MRFG3','CASH3','BEEF3','MOVI3','MRVE3','MULT3','PCAR3',
# 			'PETR3','PETR4','PRIO3','PETZ3','PSSA3','POSI3','QUAL3','RADL3','RAPT4',
# 			'RDOR3','RAIL3','SBSP3','SAPR11','SANB11','CSNA3','SULA11','SUZB3','TAEE11',
# 			'TASA4','VIVT3','TIMS3','TOTS3','UGPA3','USIM5','VALE3','VIIA3','VBBR3',
# 			'WEGE3','YDUQ3','SMAL11','BOVA11']

#Abertos
# simbolos = ['LCAM3','MOVI3','PETR3','PETR4']

#Anti-Trader
# simbolos = ['PRIO3','LCAM3','TASA4','SIMH3','CARD3','BOVA11','BPAC11','SHUL4','MOVI3','XPBR31','SMAL11']

#Erico
simbolos = ['IVVB11','BOVA11','GOLD11','PRIO3','TASA4','RECV3','MOVI3','RENT3','AGRO3','EZTC3','EVEN3','AMBP3','BMOB3','CBAV3','SQIA3']

tabela_completa = pd.DataFrame()

############# Funções Globais   ############################
## Funcao para baixaar 1 ano de dados do yfinance usada no botao atualizar
def atualizarDados():
	file1 = open('logAtualizacao.txt','w')
	
	hoje = datetime.datetime.now()
	inicio = hoje - datetime.timedelta(days = 365)
	ontem = hoje - datetime.timedelta(days = 1)

	falha = 0
	#WORKING Apenas atualizar se o arquivo de log estiver com a data diferente ou com erros
	file1.write(str(hoje) + '\n')

	#Atualizando os csv com os dados do yahoo finance
	for sim in simbolos:
		try:
			#Obter dados apenas dos ativos com erro ou data diferente de hoje
			print('Obtendo dados de ' + sim)			
			dataSimbol = ff.dados(sim + '.SA',
							inicio,
							ontem,
							True)
			print(sim + ' index ' + str(dataSimbol.index.size))
			if dataSimbol.index.size < 200: simbolos.remove(sim)
			file1.write(sim + ': Ataulizado\n')

		except:
			file1.write(sim + ': Falhou\n')
			falha += 1
	file1.close()

	# WORKING criar os dados do heatmap
	return [hoje,falha]

############# Contructors das Areas do HTML   ############################
def build_area_1():
	# data = pd.read_csv("momentum.csv")
	# data1 = data.head(10)

	return(
		#Criando tabela Top 10 / Nota
        html.Div([
        	html.Button('Atualizar', id='atual', n_clicks=0),
        	html.Div(id='atual-data',children='ultimo:'),
        	#WORKING Fazer o callback do botão de log abrir um popup com o arquivo de log
        	html.Button('Log WORKING', id='open-log', n_clicks=0),
        	html.Button('Grafico HeatMap', id='heatmap', n_clicks=0),
        	html.Button('Grafico Residuo', id='longshort', n_clicks=0),
        	html.Button('Grafico Dispersão', id='dispersao', n_clicks=0),
            dcc.Dropdown(
                id='paper1',
                options=[{'label': i, 'value': i} for i in simbolos],
                value='BOVA11'
            ),
            dcc.Dropdown(
                id='paper2',
                options=[{'label': i, 'value': i} for i in simbolos],
                value='TASA4'
            ),
            html.Div(id='atual_dados',children='Dados',style={'whiteSpace': 'pre-line'}),
         #    dcc.Interval(
         #    id='interval-component',
         #    interval=1*10000, # in milliseconds
         #    n_intervals=0
        	# )

            #WORKING a partir da seleção do dropdown atualizar a tabela
            # dcc.Dropdown(
            #     id='coluna',
            #     options=[{'label': i, 'value': i} for i in data.columns],
            #     value='colunaTOP10'
            # ),
            #WORKING Atualizando a tabela com Momentum, mas deverá atualizar a tabela TOP10
            # do Dropdown
      #       dash_table.DataTable(
    		# id='table',
		    # columns=[{"name": i, "id": i} for i in data.columns],
		    # data=data1.to_dict('records'))
		    # data=data
            # ,style={
            # 	'float': 'left',
            # 	'width':'150px'
        	# }
        ])
	)

def build_area_2():
	#WORKING Abrir o grafico de heatmap
	
	data = pd.read_csv('BOVA11.SA.csv')
	
	fig = go.Figure(go.Candlestick(x= data["Date"],
                        open= data["Open"],
                        high= data["High"],
                        low= data["Low"],
                        close= data["Close"]
                    ))

	fig.update_layout(
    title='BOVA11'),
    
	return (dcc.Graph(id='stockgraphic', figure = fig))

def build_area_3():
	#Criar a tabela do pair com os dados de periodos 80-100-120-140-160-180 dias com as colunas
	#Intervalo - ADF - ADF % - Pearson - Spearmann - RANSAC Score - RANSAC ADF - Angulo - Meia Vida
	# ADF% é o quanto menor é o ADF dos limites criticos 99%, 95% ou 90%

	# completa = pd.read_csv('tabela_completa.csv')
	return (
		html.Div(dash_table.DataTable(
    			id='tabelaCompleta',
		    	columns=[{"name": i, "id": i} for i in tabela_completa.columns],
		    	data=tabela_completa.to_dict('records'))
		))
	return html.Div(id='texto',children='texto'),
############# Pagina e definção layout   ############################

app = dash.Dash(__name__)

app.layout = html.Div(children=[
     # first row
    html.Div(children=[
        html.Div(children=[build_area_1()]
        	# html.P('Area 1 Table Top10')],
        	# html.Div(children=[build_area_1()]          
            , style ={
            	'display': 'inline-block',
            	'vertical-align': 'top',
            	'width':'200px'
            	}
            )
            ,
        html.Div(children=[build_area_2()
            # html.P('Area 2 Grafico')
            # html.Div(children=[build_area_2()])
            ],style={
            	'display': 'inline-block',
            	'vertical-align': 'top',
            	'width':'750px'
            	}
            )
        ])
		,	
        # Second row
        html.Div(children=[build_area_3()]
        	,style={
            	'float': 'left',
            	'width':'800px'
        	}
        )
    ])


################### CALLBACKs  ######################
# @app.callback(Input('interval-component','n_intervals'))
# def update_interval(n_intervals):
# 	print(n_intervals)

# Callback Botão atualizar Area 1
@app.callback(
    Output('atual-data','children'),
    Input('atual','n_clicks'),
)
def update_output(n_clicks):
	if n_clicks == 0:
		return 'Não atualizado'
	else:
		retorno = atualizarDados()
		return str(retorno[0])[:10] + ' com "{}" falhas'.format(retorno[1]) 

#Ao clicar em uma linha da Tabela Completa mostrar no grafico
@app.callback(
		Output('stockgraphic', 'figure'),
		Input('heatmap', 'n_clicks'),
		Input('longshort', 'n_clicks'),
		Input('dispersao', 'n_clicks'),
		Input('paper1', 'value'),
		Input('paper2', 'value'),
		Input('tabelaCompleta', 'active_cell')
)
def update_graphs(heatmap,longshort,dispersao,stock1,stock2,active_cell):

	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = 'heatmap'
	else:
		button_id = ctx.triggered[0]['prop_id'].split('.')[0]
		print(button_id)

	
	# FUNCIONANDO ####### Plotar o heat map  ##############
	tabela = pd.DataFrame()
	for sim in simbolos:
		dados = pd.read_csv(sim + '.SA.csv')
		tabela[sim]=dados['Close']

	if button_id in ['heatmap']:
		correlacao = tabela.corr()
		correlacao = correlacao.round(2)

		fig = px.imshow(correlacao,text_auto=True, aspect="auto",color_continuous_scale=['#006600','#fff333'])
		
	
	if button_id in ['tabelaCompleta','longshort','paper1','paper2'] :
		inter = int(active_cell['row_id']) if active_cell else 90

		data1 = pd.read_csv(stock1 + ".SA.csv")
		data2 = pd.read_csv(stock2 + ".SA.csv")

		ativo1 = data1['Close']
		ativo2 = data2['Close']

		Ativo1strip = ativo1.iloc[-inter:]
		Ativo2strip = ativo2.iloc[-inter:]

		valores = Ativo1strip/Ativo2strip
		y_est = ff.linearRegressao(valores)
		residuo = ff.residuo(valores,y_est[0])
		
		fig = go.Figure()
		### Working ### Grafico de Reiduo
		x = np.linspace(1, len(residuo[1]), len(residuo[1]))
		# print((np.zeros(len(residuo))+2))
		fig.add_trace(go.Scatter(x=x, y=residuo[1],mode='lines+markers+text',textposition='top left',))
		fig.add_trace(go.Scatter(x=x, y=(np.zeros(len(residuo[1]))+2),mode='lines', line=dict(color='#EF553B',width=1),line_dash='dash'))
		fig.add_trace(go.Scatter(x=x, y=(np.zeros(len(residuo[1]))-2),mode='lines', line=dict(color='#EF553B',width=1),line_dash='dash'))

	if button_id in ['dispersao'] :
		# Working ####### Plotar Scatter  ##############

		# inter = int(active_cell['row_id']) if active_cell else 90
		inter = 210

		data1 = pd.read_csv(stock1 + ".SA.csv")
		data2 = pd.read_csv(stock2 + ".SA.csv")

		ativo1 = data1['Close']
		ativo2 = data2['Close']

		Ativo1strip = ativo1.iloc[-inter:]
		Ativo2strip = ativo2.iloc[-inter:]

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=ativo2.iloc[-1:], y=ativo1.iloc[-1:],
                    mode='markers', marker=dict(color='green')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-90:-2], y=ativo1.iloc[-90:-2],
                    mode='markers', marker=dict(color='gold')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-110:-90], y=ativo1.iloc[-110:-90],
                    mode='markers', marker=dict(color='#ff1010')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-130:-110], y=ativo1.iloc[-130:-110],
                    mode='markers', marker=dict(color='#ff7474')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-150:-130], y=ativo1.iloc[-150:-130],
                    mode='markers', marker=dict(color='#ffacac')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-170:-150], y=ativo1.iloc[-170:-150],
                    mode='markers', marker=dict(color='#bac2ff')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-190:-170], y=ativo1.iloc[-190:-170],
                    mode='markers', marker=dict(color='#5e71ff')
                    ))
		fig.add_trace(go.Scatter(x=ativo2.iloc[-210:-190], y=ativo1.iloc[-210:-190],
                    mode='markers', marker=dict(color='#001efe')
                    ))

		a, b = np.polyfit(Ativo2strip, Ativo1strip, deg=1)
		print('a reta = ' + str(a))
		print('b reta = ' + str(b))

		y_est = a * Ativo2strip + b
		fig.add_trace(go.Scatter(x=Ativo2strip, y=y_est,
                    mode='lines', line=dict(color='red')
                    ))


	return fig

#Callback para atualizar a tabela da area 3
@app.callback(
    [Output("tabelaCompleta", "data"),
    Output("tabelaCompleta", "columns")],
    Input('longshort', 'n_clicks'),
	Input('paper1', 'value'),
	Input('paper2', 'value')
)
def atualizaTabela(n_clicks,stock1,stock2):

	data1 = pd.read_csv(stock1 + ".SA.csv")
	data2 = pd.read_csv(stock2 + ".SA.csv")
	tamanho = [90,110,130,150,170,190,210]

	# print(data1['Close'])



	tabela_completa = ff.LongShortTable(data1['Close'],data2['Close'],tamanho)

	# tabela_completa = tabela_completa.round(2)
	# print(tabela_completa.index)
	tabela_completa['id'] = tabela_completa.index
	cols = tabela_completa.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	tabela_completa = tabela_completa[cols]
	# print(tabela_completa)

	columns=[{"name": i, "id": i} for i in tabela_completa.columns]
	data=tabela_completa.to_dict('records')

	return [data,columns]

@app.callback(
    Output('atual_dados','children'),
    Input('paper1', 'value'),
	Input('paper2', 'value')
	)
def update_dados(stock1, stock2):
	data1 = pd.read_csv(stock1 + ".SA.csv")
	data2 = pd.read_csv(stock2 + ".SA.csv")

	info1 = ff.stockinfo(stock1 + ".SA")
	info2 = ff.stockinfo(stock2 + ".SA")

	print(info1['shortName'])

	ultimo1 = data1['Close'].iloc[-1]
	ultimo2 = data2['Close'].iloc[-1]

	valor = ultimo1/ultimo2
	valorAtual = info1['ask']/info2['ask']

	return ('ULTIMO DIA \n\n'+
			stock1 + ': ' + str(ultimo1.round(2)) + '\n'+
			stock2 + ': ' + str(ultimo2.round(2)) + '\n'+
			'ratio' + ': ' + str(valor.round(2))+ '\n\n'
			'ATUAL \n\n'+
			stock1 + ': ' + str(info1['ask']) + '\n'+
			stock2 + ': ' + str(info2['ask']) + '\n'+
			'ratio' + ': ' + str(valorAtual)+ '\n'

	)

################### inicializador  ######################
if __name__ == "__main__":
    app.run_server(debug=True)