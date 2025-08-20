from collections import defaultdict
import enum
from math import pi
import os
from flask import Flask, request, jsonify, render_template_string
import numpy as np
from scipy.spatial import Voronoi
import matplotlib

import MyVoronoi
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from voronoi_functions import generate_voronoi_data, generate_voronoi_plot
from voronoi_functions_clean import generate_voronoi_data_clean, generate_voronoi_plot_clean



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
        .container { display: flex; gap: 10px; }
        .form-container { flex: 1; }
        .diagram-container { flex: 2; }
        .diagram { max-width: 100%; }
        input { width: 100px; }
        button { padding: 8px 16px; }
        .form-container div {
            margin: 10px 0;
        }

        .form-container label {
            display: inline-block;
            min-width: 150px; /* Reduced from 200px */
            margin-right: 10px;
            vertical-align: middle;
        }

        .form-container input[type="number"] {
            width: 80px; /* Fixed width for number input */
            display: inline-block;
            vertical-align: middle;
        }
        
        .form-container input[type="checkbox"] {
            width: auto;
            margin-left: 0;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <h1>Voronoi Diagram Generator</h1>
    <div class="container">
        <div class="form-container">
            <form id="voronoiForm">
                <div>
                    <label for="points">Number of Points:</label>
                    <input type="number" id="points" name="points" min="4" max="100" value="10">
                </div>
                <div>
                    <label for="showNumbers">Display Vertex/Points Labels:</label>
                    <input type="checkbox" id="showNumbers" name="showNumbers" checked>
                </div>
                <div>
                    <label for="showPoints">Show Points:</label>
                    <input type="checkbox" id="showPoints" name="showPoints" checked>
                </div>
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
            const showNumbers = document.getElementById('showNumbers').checked;
            const showPoints = document.getElementById('showPoints').checked;
            const response = await fetch(`/generate?points=${points}&show_numbers=${showNumbers}&show_points=${showPoints}`);
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
    diagram_data = generate_voronoi_data_clean(10)
    
    return render_template_string(HTML_TEMPLATE, diagram_data=diagram_data)

@app.route('/generate')
def generate():
    input_points = int(request.args.get('points', 10))
    show_numbers = request.args.get('show_numbers', 'true').lower() == 'true'
    show_points = request.args.get('show_points', 'true').lower() == 'true'
    
    # # Generate the diagram with number display option
    # count = 0
    # while True:
    #     try:
    #         # your main code here
    #         diagram_data_clean = generate_voronoi_data_clean(input_points, show_numbers=show_numbers)
    #         print("RUNNING", count)
    #         count += 1
    #         #break
    #     except Exception as e:
    #         print(f"Error occurred: {e}")
    #         break

    diagram_data_clean = generate_voronoi_data_clean(input_points, show_numbers=show_numbers, show_points=show_points)

    
    return jsonify({'diagram': diagram_data_clean})

if __name__ == '__main__':
    from flask_cors import CORS
    CORS(app)
    app.run(debug=True, host='127.0.0.1', port=5002)
