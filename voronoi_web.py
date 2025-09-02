from collections import defaultdict
import enum
from math import pi
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
from scipy.spatial import Voronoi
import matplotlib
import json
matplotlib.use('Agg')  # Set matplotlib to non-interactive mode
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from voronoi_functions_clean import generate_voronoi_data_clean



# Set matplotlib backend explicitly
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__, static_folder='static')

# Serve the main page
@app.route('/')
def index():
    return app.send_static_file('index.html')




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

    diagram_data_clean, json_str = generate_voronoi_data_clean(input_points, show_numbers=show_numbers, show_points=show_points)
    json_data = json.loads(json_str)
    with open('voronoi_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print("JSON STR:", json_str)
    i = 0
    for point in json_data['points']:
        print("POINT #", i, ":", point)
        i += 1
    i = 0
    for vertex in json_data['vertices']:
        print("VERTEX #", i, " : ", vertex)
        i += 1
    for ridge in json_data['ridges']:
        print("RIDGE:", ridge[0], '-', ridge[1])
    for region in json_data['regions']:
        print("REGION #", region, " : ", json_data['regions'][region])

    return jsonify({'diagram': diagram_data_clean, 'json_str': json_str})

if __name__ == '__main__':
    from flask_cors import CORS
    CORS(app)
    app.run(debug=True, host='127.0.0.1', port=5002)
