# Web Application Report for Machine Learning Algorithms
![markdown](/static/web.png)

## Introduction
This project was an exercise defined by Dr. Taj Bakhsh for students in the master's software engineering field of Urmia University

The purpose of this exercise is to use ready-made tools to visually display the graph of the following algorithms. To implement this site and the algorithms, we need to write the front-end and back-end code for the following machine learning algorithms:

- Decision Tree
- SVM
- KNN
- Kmeans

## Front End Section

To start this exercise, a `main.html` page was created. In the initial section, the header includes a menu with the following elements:

- **Home**
- **A drop-down menu** of algorithms that redirects the user to the page corresponding to each algorithm when selected.
- **About Us Page**
- **Contact Us Page**

A slider was created using JavaScript and styled with CSS. Images of the algorithms in question were uploaded to these slides. In the next section, boxes were designed to explain each algorithm, which include images and descriptive text. Finally, a footer was designed for the **Data Schools** site.

The user can visually see how all four algorithms work through this web environment. For each algorithm, an HTML page is created and styled with CSS. These templates receive input from the user according to the algorithms implemented in the Flask file and then plot and display the corresponding graph.

## Back End Section

To implement the back-end section of this AI site, Flask has been used. More explanations will be provided below.

### Project Structure

**Folders & Files:**

- `static/`: Includes static files such as images, CSS files, and JavaScript files.
- `templates/`: Includes HTML templates for web pages.
- `app.py`: The primary file containing Python and Flask code.
# Libraries and Packages Used
```python

from flask import Flask, render_template, request, send_from_directory
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn. tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib

```
- **Flask:** A simple and flexible web framework for implementing web applications in Python. Flask allows for the easy creation and running of web applications using the MVC (Model-View-Controller) architecture.

- **NumPy and Pandas:** Two of the most widely used libraries in Python for working with numerical data and tabular data structures (DataFrames). These libraries facilitate data preparation and computational operations on numerical data.

- **Scikit-learn:** A popular library for machine learning in Python that includes a set of machine learning algorithms and their related tools. Scikit-learn enables easy training and usage of machine learning models.

- **Matplotlib:** A powerful library for drawing charts and images in Python. This library allows for the easy graphical display of data and the creation of visually appealing images.

- **io and base64:** These libraries are used to work with binary data and convert it to the Base64 format. This process allows for the conversion of binary-based images into a web-viewable format.

- **matplotlib.use('Agg'):** This setting changes the graphical backend of the matplotlib package to 'Agg'. 'Agg' is a non-interactive backend used to generate images without the need for a graphical window. This change is typically made in web applications to enable the generation and display of images non-interactively, without the necessity of opening a graphical window.

## App Setup

The app launches using Flask and downloads static files and HTML templates.
```python
app = Flask(__name__, static_url_path='/static', static_folder='static')
```
## Loading Iris Dataset

The Iris dataset is loaded, and the first two features are selected for analysis.

```python
# Load iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Use only the first two features
y = iris.target
```
## Paths and Routes
Home page that renders the main.html  template.

```python

@app.route('/')
def main():
    return render_template('main.html')

```
# About Us & Contact Us Pages
Simple pages to display team information and contact information.

```python
@app.route('/about_us')
def about_us_page():
    return render_template('about-us.html')

@app.route('/contact_us')
def contact_us_page():
    return render_template('contact-us.html')

```
## Importing Images
This path is used to import image files from the static  folder.
```python
app.route('/img/<path:filename>')
def serve_image(filename):
    return send_from_directory('static', filename)

```
## Training Models

- **train_logistic_regression():** Teaching the logistic regression model.
- **train_svm():** Support Vector Machine Model (SVM) Training.
- **train_decision_tree():** Decision Tree Model Training.
- **train_knn():** Nearest Neighbors Model (KNN) Training.
- **train_kmeans():** K-Means clustering model training.

### SVM Model Training
```python
def train_svm(kernel='linear', C=1.0, degree=3):
    svm_model = SVC(kernel=kernel, C=C, degree=degree if kernel == 'poly' else 3)
svm_model.fit(X, y)
    return svm_model
```

This function trains an SVM model using the kernel parameters **(kernel type)**, C **(regularization parameter)**, and degree **(polynomial degree for polynomial core)**. If the kernel type  is not poly, the  degree value is set to 3.  The fit function  trains the model and returns the trained model. 




## Decision Tree Model Training
```python
def train_decision_tree(max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth)
clf.fit(X, y)
    return clf
```
This function trains a Decision Tree model using the max_depth parameter **(maximum tree depth)**.  The X and y  data are used to train the model.  The fit  function trains the model and returns the trained model.

## KNN Model Training 
```python
def train_knn(n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X, y)
    return knn
```

This function trains a KNN  model using the n_neighbors parameter (number of neighbors).  The fit  function trains the model and returns the trained model.

## Training the K-Means Model
```python
def train_kmeans(n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)
    return kmeans
```

This function trains a K-Means  model using the n_clusters parameter (number of clusters).  The fit  function trains the model and returns the trained model.
# Drawing Functions 
## Drawing the Decision Tree
![markdown](/static/tree.png)
```python
def plot_decision_tree(clf):
    plt.figure(figsize=(25, 12))
    plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names)
 buffer = BytesIO()
 plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
plt.close()
    return plot_data
```

This function maps the CLF decision tree model. The plot_tree  function  uses sklearn to draw the tree. The generated image is returned as a Base64  string, which is suitable for display on web pages.
 
## Drawing the KNN 
![markdown](/static/knn.png)
```python
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

```
This function plots the  KNN  model based on the input features data  . First, a grid of points in the feature space is created, and then the model prediction is made for each point of this network to determine the model decision boundary. The data points and the model decision boundary are drawn and the final image is returned as a  Base64  string.




## Drawing the K-Means 
![markdown](/static/kmean.png)
```python
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
```
This function plots the  K-Means  model based on the input features data  . First, the data points and labels of the clusters are drawn, and then the center of the clusters (centroids) are displayed. The final image is returned as  a Base64  string.
 
## Drawing the SVM Decision Boundary 
- Follow plot_svm_decision_boundary
This function is used to draw the decision boundary of an SVM model  with a linear core.
![markdown](/static/svm-linear.png)

```python
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
```
This function first creates a lattice of points in the feature space, and then the model prediction is made for each point of this lattice to determine the model's decision boundary. Data points, model decision boundaries, and support vectors  are plotted. The final image is returned as a  Base64  string.



- Follow plot_rbf_svm_decision_boundary
This function is used to draw the decision boundary of an SVM  model with  the RBF kernel
![markdown](/static/svm-rbf.png)

```python
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
```
The function of this function is similar to the previous one, except that this function is used for models with RBF kernels. The decision boundary, data points, and supporting vectors are drawn, and the result  is returned as a base64 string.


- Follow plot_poly_svm_decision_boundary
This function is used to draw the decision boundary of an SVM  model with a polynomial core.
![markdown](/static/pol.png)
```python
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
```
This function also works similarly to the previous functions, except that it is used for models with polynomial kernels. The decision boundary, data points, and supporting vectors are drawn, and the result is returned as  a base64  string.
As a result, these functions allow users to visualize the decision boundary of  SVM  models with different cores. This helps to better understand how SVM  models work and the impact of different parameters on these models. Users can gain a deeper understanding of these models by changing the parameters and observing changes in the decision boundary. This feature is especially useful in education. And learning machine learning concepts is very useful.


## User Interface Functions
### Decision Tree
```python
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
```

This function creates a web page to get the max_depth parameter  from the user and draw the decision tree. If the form is submitted,  the max_depth  parameter is received and the model is trained and drawn. The drawing result is displayed as  a base64  image on the web page.

### Support Vector Machine (SVM)
```python
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
```



This function creates a web page to get kernel, C, and degree  parameters from the user and draw the SVM model decision boundary. If the form is submitted, the relevant parameters are received and the model is trained. Then, based on the kernel type, the model decision boundary is drawn and the result is displayed as a  base64  image on the web page.

### Nearest Neighborhood (KNN)
```python
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
```
This function creates a web page to get the n_neighbors parameter  from the user and draw the KNN model diagram. If the form is submitted,  the n_neighbors  parameter is received and the model is trained. Then the  KNN  model is drawn and the result is displayed as  a Base64  image on the web page.

 ### K-Means Clustering
 ![markdown](/static/kmean.png)
```python
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
```
This function creates a web page to get the n_clusters parameter  from the user and plot the  K-Means  model. If the form is submitted, the n_clusters parameter  is received and the model is trained. The K-Means  model is then drawn and the result is displayed as a  base64  image on the web page.

### Server Setup
```python
if __name__ == '__main__':
    app.run(debug=True, port=8080)


  if __name__ == '__main__'
```
-	This line of code checks whether the script has been executed directly. If the script has been executed directly, the value of  the __name__  will be equal to '__main__'. This is a standard feature in the Python programming language that allows programmers to determine whether a Python file has been executed directly or imported as a module.
-	If the __name__  value is equal to '__main__',  the code block below it will be executed. If this file is imported as a module, this block will not run.
  app.run(debug=True, port=8080):
-	This line of code will launch the Flask  app.
-	app.run()  is the Flask  method that launches the web server.
-	debug=True: This parameter activates debugging mode. When debugging mode is enabled, Flask automatically restarts the server whenever there is a change in the code. Also, more information will be displayed in the terminal if there is an error. 
-	port=8080: This parameter sets the port used for the server. By default, Flask uses port 5000, but setting this parameter can specify another port such as 8080 for the server.


### Conclusion:
This web application has been created using the  Flask  framework and various machine learning libraries in Python. Users can enter different parameters of models through the web interface and view the results visually. This program allows users to get acquainted with different machine learning concepts interactively and see the impact of different parameters on the models.
 
