from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo entrenado
with open('modelo_aprobacion.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Página principal - Dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para obtener resultados existentes
@app.route('/get-results')
def get_results():
    try:
        df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.xlsx'))
        
        return jsonify({
            'total': len(df),
            'aprobados': int(df['Predicción'].sum()),
            'no_aprobados': len(df) - int(df['Predicción'].sum()),
            'stats': {
                'studytime': df['studytime'].mean(),
                'failures': df['failures'].mean(),
                'absences': df['absences'].mean(),
                'G1': df['G1'].mean()
            }
        })
    except:
        return jsonify({'error': 'No hay datos disponibles'}), 404

# Endpoint para procesar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
        
    if file and file.filename.endswith('.xlsx'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Procesar archivo
            df = pd.read_excel(filepath)
            features = ['studytime', 'failures', 'absences', 'G1', 'G2']
            
            # Verificar columnas
            missing = [col for col in features if col not in df.columns]
            if missing:
                return jsonify({'error': f"Faltan columnas: {', '.join(missing)}"}), 400
            
            # Realizar predicciones
            df['Predicción'] = modelo.predict(df[features])
            df['Probabilidad'] = modelo.predict_proba(df[features])[:, 1].round(2)
            
            # Guardar resultados
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.xlsx')
            df.to_excel(results_path, index=False)
            
            # Calcular estadísticas
            stats = {
                'studytime': df['studytime'].mean().round(1),
                'failures': df['failures'].mean().round(1),
                'absences': df['absences'].mean().round(1),
                'G1': df['G1'].mean().round(1)
            }
            
            return jsonify({
                'success': True,
                'total': len(df),
                'aprobados': int(df['Predicción'].sum()),
                'no_aprobados': len(df) - int(df['Predicción'].sum()),
                'stats': stats,
                'accuracy': 75  # Puedes reemplazar esto con la precisión real de tu modelo
            })
            
        except Exception as e:
            return jsonify({'error': f"Error al procesar el archivo: {str(e)}"}), 500
    
    return jsonify({'error': 'Formato de archivo no válido. Use .xlsx'}), 400

# Endpoint para generar gráficos
@app.route('/generate-charts')
def generate_charts():
    try:
        df = pd.read_excel(os.path.join(app.config['UPLOAD_FOLDER'], 'resultados.xlsx'))
        
        # Gráfico 1: Distribución de aprobados
        fig1 = px.pie(
            df, 
            names='Predicción', 
            title='Distribución de Resultados',
            labels={'0': 'No Aprobado', '1': 'Aprobado'},
            color='Predicción',
            color_discrete_map={'0': '#EF4444', '1': '#10B981'}
        )
        
        # Gráfico 2: Probabilidad por estudiante
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df.index,
            y=df['Probabilidad'],
            marker_color=df['Probabilidad'].apply(lambda x: '#10B981' if x > 0.5 else '#EF4444'),
            name='Probabilidad'
        ))
        fig2.update_layout(
            title='Probabilidad de Aprobación por Estudiante',
            xaxis_title='Estudiantes',
            yaxis_title='Probabilidad',
            showlegend=False
        )
        
        # Convertir gráficos a HTML
        chart1 = fig1.to_html(full_html=False)
        chart2 = fig2.to_html(full_html=False)
        
        return jsonify({
            'success': True,
            'chart1': chart1,
            'chart2': chart2,
            'stats': {
                'A': df['studytime'].mean().round(1),
                'B': df['failures'].mean().round(1),
                'C': df['absences'].mean().round(1),
                'D': df['G1'].mean().round(1),
                'S': len(df),
                'E': df['Predicción'].sum()
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'No hay datos disponibles o ocurrió un error'
        }), 500

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)