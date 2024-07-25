from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados
ordinal_encoder = joblib.load('modelo_ordinalEncoder.pkl')
scaler = joblib.load('modelo_StandarScaler.pkl')
pca = joblib.load('modelo_PCA.pkl')
model = joblib.load('modelo_RandomForest.pkl')
app.logger.debug('Modelos cargados correctamente.')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Obtener los datos enviados en el request
            habilidad_lectura = request.form['habilidad_lectura']
            habilidad_escritura = request.form['habilidad_escritura']
            habilidad_matematicas = request.form['habilidad_matematicas']
            participacion = request.form['participacion']
            comportamiento = request.form['comportamiento']

            # Crear un DataFrame con los datos
            data_df = pd.DataFrame([[habilidad_lectura, habilidad_escritura, habilidad_matematicas, participacion, comportamiento]],
                                   columns=['habilidad_lectura', 'habilidad_escritura', 'habilidad_matematicas', 'participacion', 'comportamiento'])
            app.logger.debug(f'DataFrame creado: {data_df}')

            # Preprocesar los datos
            data_df[['habilidad_lectura', 'habilidad_escritura', 'habilidad_matematicas', 'participacion', 'comportamiento']] = ordinal_encoder.transform(data_df[['habilidad_lectura', 'habilidad_escritura', 'habilidad_matematicas', 'participacion', 'comportamiento']])
            data_df = scaler.transform(data_df)
            
            data_df = pca.transform(data_df)
            app.logger.debug(f'Datos preprocesados: {data_df}')

            app.logger.debug(f'Datos preprocesados: {data_df}')

            # Realizar predicciones
            prediction = model.predict(data_df)[0]
            app.logger.debug(f'Predicción: {prediction}')
        except Exception as e:
            app.logger.error(f'Error en la predicción: {str(e)}')
            prediction = None

    return render_template('formulario.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
