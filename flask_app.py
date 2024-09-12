from flask import Flask, render_template, request, send_from_directory
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib

# Use 'Agg' backend for matplotlib
matplotlib.use('Agg')

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Load iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features
y = iris.target

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/about_us')
def about_us_page():
    return render_template('about-us.html')

@app.route('/contact_us')
def contact_us_page():
    return render_template('contact-us.html')

@app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)

# Train classifiers with additional parameters
def train_logistic_regression(learning_rate=0.1, regularization='l2', regularization_rate=1.0):
    if regularization == 'l1':
        clf = LogisticRegression(penalty='l1', solver='saga', C=regularization_rate, max_iter=1000)
    elif regularization == 'l2':
        clf = LogisticRegression(penalty='l2', C=regularization_rate, max_iter=1000)
    else:
        clf = LogisticRegression(C=regularization_rate, max_iter=1000)
    clf.fit(X, y)
    return clf

def train_svm(kernel='linear', C=1.0, degree=3):
    svm_model = SVC(kernel=kernel, C=C, degree=degree if kernel == 'poly' else 3)
    svm_model.fit(X, y)
    return svm_model


def train_decision_tree(max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)
    return clf

def train_knn(n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    return knn

def train_kmeans(n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans

# Plot functions
def plot_decision_tree(clf):
    plt.figure(figsize=(25, 12))
    plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def plot_knn(knn, features):
    plt.figure(figsize=(10, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.scatter(features[0][0], features[0][1], color='red', marker='x', s=100, label='Input Point')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def plot_kmeans(kmeans, features):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='red', label='Centroids')
    plt.scatter(features[0][0], features[0][1], color='blue', marker='o', s=100, label='Input Point')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def plot_svm_decision_boundary(svm_model, kernel, C, feature1, feature2):
    # Create a mesh grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and support vectors
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.scatter(feature1, feature2, color='red', marker='x', s=100, label='Input Point')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary ({kernel} Kernel)')
    plt.legend()
    
    # Save plot as base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def plot_rbf_svm_decision_boundary(svm_model, feature1, feature2):
    # Create a mesh grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and support vectors
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.scatter(feature1, feature2, color='red', marker='x', s=100, label='Input Point')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary (RBF Kernel)')
    plt.legend()
    
    # Save plot as base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

    # Save plot as base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data
def plot_poly_svm_decision_boundary(svm_model, degree, C, feature1, feature2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.scatter(feature1, feature2, color='red', marker='x', s=100, label='Input Point')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'SVM Decision Boundary (Polynomial Kernel, degree={degree}, C={C})')
    plt.legend()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data


@app.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree_page():
    if request.method == 'POST':
        try:
            max_depth = int(request.form['max_depth'])  # Get max depth from user input
        except ValueError:
            return "Invalid input. Please enter an integer value."
        
        clf = train_decision_tree(max_depth=max_depth)  # Pass max depth to training function
        plot_data = plot_decision_tree(clf)
        return render_template('decision_tree.html', plot_data=plot_data)
    return render_template('decision_tree.html')

@app.route('/svm', methods=['GET', 'POST'])
def svm_page():
    if request.method == 'POST':
        try:
            kernel = request.form['kernel']
            C = float(request.form['C'])
            degree = int(request.form.get('degree', 3))
        except ValueError:
            return "Invalid input. Please enter numeric values."
        
        clf = train_svm(kernel=kernel, C=C, degree=degree)
        features = [[5, 2]] 
        
        if kernel == 'linear':
            plot_data = plot_svm_decision_boundary(clf, kernel, C, features[0][0], features[0][1])
        elif kernel == 'rbf':
            plot_data = plot_rbf_svm_decision_boundary(clf, features[0][0], features[0][1])
        elif kernel == 'poly':
            plot_data = plot_poly_svm_decision_boundary(clf, degree, C, features[0][0], features[0][1])
        else:
            return "Invalid kernel type."
        
        return render_template('svm.html', plot_data=plot_data)
    return render_template('svm.html')



@app.route('/knn', methods=['GET', 'POST'])
def knn_page():
    if request.method == 'POST':
        try:
            n_neighbors = int(request.form['n_neighbors'])
        except ValueError:
            return "Invalid input. Please enter an integer value."
        
        knn_model = train_knn(n_neighbors=n_neighbors)
        # Use some meaningful input features instead of (0, 0)
        features = [[5, 2]] 
        plot_data = plot_knn(knn_model, features)
        return render_template('knn.html', plot_data=plot_data)
    return render_template('knn.html')

@app.route('/kmeans', methods=['GET', 'POST'])
def kmeans_page():
    if request.method == 'POST':
        try:
            n_clusters = int(request.form['n_clusters'])
        except ValueError:
            return "Invalid input. Please enter an integer value."
        
        kmeans_model = train_kmeans(n_clusters=n_clusters)
        # Use some meaningful input features instead of (0, 0)
        features = [[5, 2]]
        plot_data = plot_kmeans(kmeans_model, features)
        return render_template('kmeans.html', plot_data=plot_data)
    return render_template('kmeans.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)

