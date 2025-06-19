from collections import defaultdict
from math import pi
import os
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from voronoi_functions import generate_voronoi_data, generate_voronoi_plot, generate_voronoi_diagram

# Set matplotlib backend explicitly
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Voronoi Diagram Generator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { display: flex; gap: 20px; }
        .form-container { flex: 1; }
        .diagram-container { flex: 2; }
        .diagram { max-width: 100%; }
        input { width: 100px; }
        button { padding: 8px 16px; }
    </style>
</head>
<body>
    <h1>Voronoi Diagram Generator</h1>
    <div class="container">
        <div class="form-container">
            <form id="voronoiForm">
                <label for="points">Number of Points:</label>
                <input type="number" id="points" name="points" min="2" max="100" value="10">
                <button type="submit">Generate Diagram</button>
            </form>
        </div>
        <div class="diagram-container">
            <img id="diagram" class="diagram" src="{{ diagram_data }}" alt="Voronoi Diagram">
        </div>
    </div>
    <script>
        document.getElementById('voronoiForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const points = document.getElementById('points').value;
            const response = await fetch(`/generate?points=${points}`);
            const data = await response.json();
            document.getElementById('diagram').src = data.diagram;
        });
    </script>
</body>
</html>
'''



@app.route('/')
def index():
    # Generate initial diagram with 10 points
    diagram_data = generate_voronoi_diagram(10)
    (points, vertices, ridges) = generate_voronoi_data(10)
    diagram_data = generate_voronoi_plot(points, vertices, ridges, 1000, 1000)
    return render_template_string(HTML_TEMPLATE, diagram_data=diagram_data)

@app.route('/generate')
def generate():
    points = int(request.args.get('points', 10))
    diagram_data = generate_voronoi_diagram(points)
    (points, vertices, ridges) = generate_voronoi_data(10)
    diagram_data = generate_voronoi_plot(points, vertices, ridges, 1000, 1000)
    return jsonify({'diagram': diagram_data})

if __name__ == '__main__':
    from flask_cors import CORS
    CORS(app)
    app.run(debug=True, host='127.0.0.1', port=5000)
