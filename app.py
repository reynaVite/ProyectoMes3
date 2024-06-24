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

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Obtener los datos enviados en el request
            hp = float(request.form['HP'])
            max_speed = float(request.form['MaxSpeed'])
            cruise_speed = float(request.form['CruiseSpeed'])
            rate_of_climb_all_eng = float(request.form['RateOfClimbAllEng'])
            rate_of_climb_one_eng = float(request.form['RateOfClimbOneEng'])

            # Crear un DataFrame con los datos
            data_df = pd.DataFrame([[hp, max_speed, cruise_speed, rate_of_climb_all_eng, rate_of_climb_one_eng]],
                                   columns=['HP or lbs thr ea engine', 'Max speed Knots', 'Rcmnd cruise Knots', 'All eng rate of climb', 'Eng out rate of climb'])
            app.logger.debug(f'DataFrame creado: {data_df}')

            # Realizar predicciones
            prediction = model.predict(data_df)[0]
            app.logger.debug(f'Predicción: {prediction}')
        except Exception as e:
            app.logger.error(f'Error en la predicción: {str(e)}')
            prediction = None

    return render_template('formulario.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
