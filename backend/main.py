from flask import Flask, request, jsonify
import argparse
import os
from tomotwin.modules.common.io.mrc_format import MrcFormat
from tomotwin.modules.inference.embedor import TorchEmbedor

app = Flask(__name__)

# Global variables
tomo = None
embedor = None
window_size = 37  # Default window size, or extract from the model

def initialize_embedor(mrc_file_path, model_path):
    global embedor, tomo, window_size
    # Load the MRC file
    tomo = MrcFormat.read(mrc_file_path)

    # Initialize TorchEmbedor with the required parameters
    embedor = TorchEmbedor(
        weightspth=model_path,
        batchsize=1,  # Update as needed
        workers=0,  # Update as needed
    )
    # Optionally, extract window size from the model
    # window_size = get_window_size(model_path)

def extract_subvolume(tomo, coordinate, window_size):
    # Logic to extract the subvolume from 'tomo' based on 'coordinate'
    z, y, x = coordinate
    subvolume = tomo[z:z+window_size, y:y+window_size, x:x+window_size]
    return subvolume

def embedding_function(tomo, coordinate, embedor, window_size):
    subvolume = extract_subvolume(tomo, coordinate, window_size)
    embedding = embedor.embed(subvolume)
    return embedding

@app.route('/compute_embedding', methods=['POST'])
def compute_embedding():
    try:
        coordinate = request.json['coordinate']
        assert len(coordinate) == 3, "Coordinate must be a 3D point"
    except (KeyError, AssertionError) as e:
        return jsonify({"error": str(e)}), 400

    embedding = embedding_function(tomo, coordinate, embedor, window_size)
    return jsonify({"embedding": embedding.tolist()})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Flask app for embeddings.')
    parser.add_argument('mrc_file', help='Path to the MRC file')
    parser.add_argument('model_path', help='Path to the model weights file')
    args = parser.parse_args()

    # Initialize embedor with MRC file and model path
    initialize_embedor(args.mrc_file, args.model_path)

    app.run(debug=True, port=5000)
