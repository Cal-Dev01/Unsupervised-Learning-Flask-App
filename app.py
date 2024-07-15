from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import pandas as pd
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
data_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'all_data.csv')
graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'scatter_plot.png')

# Ensure the uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def save_data(data):
    """Save new data to the existing data file."""
    if os.path.exists(data_file_path):
        existing_data = pd.read_csv(data_file_path)
        updated_data = pd.concat([existing_data, data], ignore_index=True)
    else:
        updated_data = data
    updated_data.to_csv(data_file_path, index=False)


def train_model():
    """Train the KMeans model using the data from the data file and perform analysis."""
    data = pd.read_csv(data_file_path)
    X = data[['size', 'sound']]

    # Train the KMeans model
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Generate and save the graph with clusters
    plt.figure()
    plt.scatter(data['size'], data['sound'], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
    plt.xlabel('size')
    plt.ylabel('sound')
    plt.title('KMeans Clustering of Size vs Sound')
    plt.savefig(graph_path)
    plt.close()

    # Analysis
    cluster_counts = pd.Series(labels).value_counts().to_dict()
    return cluster_counts, centroids


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        data = pd.read_csv(file)

        # Check if required columns are in the data
        if 'size' in data.columns and 'sound' in data.columns:
            save_data(data)
            cluster_counts, centroids = train_model()
            return jsonify({
                'message': 'Data uploaded and clusters generated successfully!',
                'graph_url': url_for('get_graph'),
                'cluster_counts': cluster_counts,
                'centroids': centroids.tolist()
            })
        else:
            return jsonify({'error': 'CSV file must contain size and sound columns'}), 400


@app.route('/graph')
def get_graph():
    return send_file(graph_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
