import pandas as pd

# Datos de ejemplo
datos = {
    'studytime': [2, 3, 1, 4, 2],
    'failures': [0, 1, 2, 0, 3],
    'absences': [4, 10, 2, 0, 15],
    'G1': [12, 8, 10, 15, 6],
    'G2': [13, 9, 11, 16, 7]
}

df = pd.DataFrame(datos)
df.to_excel('ejemplo_prediccion.xlsx', index=False)
