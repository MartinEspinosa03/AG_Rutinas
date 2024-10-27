from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import time
from Algoritmo_Genetico import AlgoritmoGenetico

app = Flask(__name__)
ag = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup', methods=['POST'])
def setup():
    data = request.json
    tipo_rutina = data['tipo_rutina']
    porcentaje_cruza = float(data['porcentaje_cruza'])
    porcentaje_mut_ind = float(data['porcentaje_mut_ind'])
    porcentaje_mut_gen = float(data['porcentaje_mut_gen'])
    poblacion_inicial = int(data['poblacion_inicial'])
    poblacion_max = int(data['poblacion_max'])
    generaciones = int(data['generaciones'])

    global ag
    selected_categories = data.get('grupos_musculares', [])
    ag = AlgoritmoGenetico(tipo_rutina, selected_categories)
    ag.setup(porcentaje_cruza, porcentaje_mut_ind, porcentaje_mut_gen, poblacion_inicial, poblacion_max, generaciones)

    return jsonify({"message": "Algoritmo configurado con éxito"})

@app.route('/results', methods=['GET'])
def results():
    if ag is None:
        return jsonify({"error": "El algoritmo no ha sido configurado"}), 400

    if not hasattr(ag, 'all_records'):
        return jsonify({"error": "El algoritmo no ha sido ejecutado"}), 400

    return jsonify(ag.all_records), 200

@app.route('/graphics/<filename>', methods=['GET'])
def serve_graphics(filename):
    return send_from_directory('graphics', filename)

@app.route('/run', methods=['GET'])
def run():
    if ag is None:
        return jsonify({"error": "Configura el algoritmo primero"}), 400

    best_exercises, best_weight = ag.run()
    graphic_url = ag.render_graphics()  
    return jsonify({
        "best_exercises": best_exercises,
        "best_weight": best_weight,
        "graphic_url": f"/graphics/{os.path.basename(graphic_url)}"
    })

if __name__ == '__main__':
    app.run(debug=True)
