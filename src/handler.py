from flask import Flask, request, Response
from healthinsurance.HealthInsurance import HealthInsurance


# Carregando o Modelo
modelo = pickle.load( open( 'src/model/model_linear_regresion.pkl', 'rb'))

# Inicializando a Instância da API
app = Flask( __name__ )

# Endpoint = /predict (acessar essa função já executa o health_insurance)
@app.route( '/predict', methods=['POST'] )

def health_insurace_predict():
    # Obtém o JSON da solicitação POST
    teste_json = request.get_json()

    if teste_json: # há dados = se houver dados
        if isinstance( teste_json, dict ): # exemplo único = eu testo o JSON para ver se é um tipo de dicionário
            teste_raw = pd.DataFrame( teste_json, index=[0] ) # cria um DataFrame e define o índice como 0

        else: # exemplo múltiplo == se ele existir, há uma linha ou várias
            teste_raw = pd.DataFrame( teste_json, columns=teste_json[0].keys() )

            # Instancia a classe HealthInsurance
            pipeline = HealthInsurance()

            # Limpeza dos dados
            df1 = pipeline.data_cleaning( teste_raw )

            # Engenharia de características
            df2 = pipeline.feature_engineering( df1 )

            # Preparação dos dados
            df3 = pipeline.data_preparation( df2 )

            # Predição
            df_resposta = pipeline.get_prediction( modelo, teste_raw, df3 )

            return df_resposta
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get( 'PORT', 5000 )
    app.run('0.0.0.0', port=port)
