from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import os
import plotly.express as px
from plotly.offline import plot
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo entrenado
with open('modelo_aprobacion.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Página principal: formulario de carga
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Leer archivo
            df = pd.read_excel(filepath)

            # Columnas necesarias
            features = ['studytime', 'failures', 'absences', 'G1', 'G2']
            missing = [col for col in features if col not in df.columns]

            if missing:
                error_msg = f"Faltan columnas necesarias en el archivo: {', '.join(missing)}"
                return render_template('index.html', error=error_msg)

            # Si todo está bien, continuamos
            df_features = df[features]
            df['Predicción'] = modelo.predict(df_features)

            # Crear gráfico interactivo con Plotly
            conteo = df['Predicción'].value_counts().rename({0: 'No Aprobó', 1: 'Aprobó'})
            conteo_df = pd.DataFrame({'Resultado': conteo.index, 'Cantidad': conteo.values})

            fig = px.bar(
                conteo_df,
                x='Resultado',
                y='Cantidad',
                color='Resultado',
                color_discrete_map={'Aprobó': 'skyblue', 'No Aprobó': 'salmon'},
                title='Resultados de la Clasificación'
            )

            # Convertir gráfico a HTML
            grafico_html = plot(fig, output_type='div')

            # Guardar resultados para exportación si se necesita
            df.to_excel('uploads/resultados.xlsx', index=False)

            return render_template(
                'resultados.html',
                tables=[df.to_html(classes='data', header="true", index=False)],
                grafico=grafico_html
            )

    # GET o en caso de error general
    return render_template('index.html')

# Ruta para descargar los resultados
@app.route('/exportar')
def exportar():
    return redirect(url_for('static', filename='uploads/resultados.xlsx'))

if __name__ == '__main__':
    app.run(debug=True)
