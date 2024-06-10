from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        age = float(request.form['Age'])
        height = float(request.form['Height'])
        weight = float(request.form['Weight'])
        FAF = float(request.form['FAF'])
        TUE = float(request.form['TUE'])
        CAEC = float(request.form['CAEC'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[age, height, weight, FAF, TUE, CAEC]],
                               columns=['Age', 'Height', 'Weight', 'FAF', 'TUE', 'CAEC'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)[0]
        app.logger.debug(f'Predicción: {prediction}')
        
        # Devolver las predicciones como respuesta JSON
        return render_template('formulario.html', prediction=prediction)
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return render_template('formulario.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
