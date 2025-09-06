import nbformat as nbf
import os

# Topics for the AI learning roadmap (22 weeks)
topics = [
    ("01_python_basics", "Python Basics", "Learn the basics of Python: variables, loops, functions, and data types."),
    ("02_numpy_pandas", "NumPy & Pandas", "Explore NumPy arrays and Pandas DataFrames for data manipulation."),
    ("03_visualization_stats", "Visualization & Statistics", "Learn how to visualize data and understand basic statistics."),
    ("04_linear_regression", "Linear Regression", "Understand and implement linear regression for prediction tasks."),
    ("05_logistic_regression", "Logistic Regression", "Use logistic regression for binary classification problems."),
    ("06_decision_trees", "Decision Trees", "Learn how decision trees split data and make predictions."),
    ("07_random_forests", "Random Forests", "Use ensembles of decision trees for better performance."),
    ("08_svms", "Support Vector Machines", "Learn about SVMs and how they separate classes with hyperplanes."),
    ("09_kmeans", "K-Means Clustering", "Cluster data points into groups with K-Means."),
    ("10_neural_networks", "Neural Networks", "Build simple feedforward neural networks."),
    ("11_cnns", "Convolutional Neural Networks", "Use CNNs for image classification tasks."),
    ("12_rnns", "Recurrent Neural Networks", "Understand RNNs for sequence modeling."),
    ("13_lstms", "Long Short-Term Memory", "Learn how LSTMs improve RNNs for longer dependencies."),
    ("14_transformers", "Transformers", "Explore transformer models for NLP."),
    ("15_nlp_basics", "NLP Basics", "Preprocess and analyze text data."),
    ("16_embeddings", "Word Embeddings", "Learn about word2vec, GloVe, and embedding spaces."),
    ("17_gans", "Generative Adversarial Networks", "Understand GANs and how they generate data."),
    ("18_rl", "Reinforcement Learning", "Explore agents, environments, and rewards."),
    ("19_end_to_end", "End-to-End ML Project", "Combine skills into a small project."),
    ("20_ai_production", "AI in Production", "Learn model saving, loading, and deployment basics."),
    ("21_project_nlp", "Mini NLP Project", "Build and evaluate an NLP pipeline."),
    ("22_capstone", "Capstone Project", "Do a full end-to-end AI project."),
]

# Detailed intros (Markdown content)
intros = {
    "Python Basics": """# Python Basics

Python is the most popular programming language for AI and data science.
It is simple, readable, and comes with powerful libraries.

**Key concepts to learn:**
- Variables and data types
- Lists, dictionaries, sets, tuples
- Loops (`for`, `while`)
- Functions and scope
- Basic error handling

**Why it matters:**
Before diving into AI, you must master the programming foundations.
""",
    "NumPy & Pandas": """# NumPy & Pandas

NumPy and Pandas are the backbone of data science in Python.

**NumPy:**
- Efficient handling of arrays and matrices
- Supports vectorized operations

**Pandas:**
- High-level data manipulation with DataFrames
- Easy filtering, grouping, and aggregation

**Why it matters:**
Most AI workflows start with cleaning and preparing data. NumPy and Pandas make this possible.
""",
    "Visualization & Statistics": """# Visualization & Statistics

Data visualization helps you **see patterns** and **communicate insights**.

**Libraries:**
- Matplotlib
- Seaborn

**Statistics basics:**
- Mean, median, mode
- Variance and standard deviation
- Probability distributions

**Why it matters:**
Good analysis requires both visualization and statistical reasoning.
""",
    "Linear Regression": """# Linear Regression

Linear regression is one of the simplest machine learning models.  
It models the relationship between input variables (X) and output variable (y) with a line.

**Why it matters:**
- Foundation of supervised learning
- Used for predicting continuous values (e.g., house prices, sales)

**Key ideas:**
- Fit the "best" line through data points
- Minimize squared errors between predictions and true values
""",
    "Logistic Regression": """# Logistic Regression

Logistic Regression is used for **binary classification** tasks.  
Instead of predicting a number, it predicts the probability of belonging to a class.

**Why it matters:**
- Used in spam detection, medical diagnosis, churn prediction
- Outputs probabilities for interpretability

**Key ideas:**
- Uses the sigmoid function to squash values into [0,1]
- Decision boundary typically at 0.5
""",
    "Decision Trees": """# Decision Trees

A Decision Tree is a model that splits data by asking questions.  
Each split creates branches until reaching a decision (leaf).

**Why it matters:**
- Easy to interpret and visualize
- Works for classification and regression
- Basis for advanced models like Random Forests

**Key ideas:**
- Nodes split data by feature thresholds
- Splitting criteria: Gini index, entropy, variance reduction
""",
    "Random Forests": """# Random Forests

A Random Forest is an **ensemble** of decision trees.  
Each tree is trained on random subsets of data and features.

**Why it matters:**
- More accurate than a single decision tree
- Reduces overfitting
- Widely used in Kaggle competitions

**Key ideas:**
- Bagging: bootstrap aggregating
- Majority voting for classification
- Averaging for regression
""",
    "Support Vector Machines": """# Support Vector Machines

Support Vector Machines (SVMs) are powerful classifiers.  
They find the **hyperplane** that best separates data into classes.

**Why it matters:**
- Effective in high-dimensional spaces
- Can use kernels for non-linear boundaries

**Key ideas:**
- Margin: distance between hyperplane and closest points
- Support vectors: critical data points defining the boundary
""",
    "K-Means Clustering": """# K-Means Clustering

K-Means is an **unsupervised learning algorithm** for grouping data.  
It assigns each data point to one of K clusters.

**Why it matters:**
- Useful for market segmentation, image compression, anomaly detection

**Key ideas:**
- Choose K (number of clusters)
- Randomly assign cluster centers
- Iterate: assign points → update centers
""",
    "Neural Networks": """# Neural Networks

Neural networks are inspired by the brain.  
They consist of layers of neurons connected by weights.

**Why it matters:**
- Backbone of modern AI
- Can approximate complex, non-linear relationships

**Key ideas:**
- Layers: input, hidden, output
- Activation functions (ReLU, sigmoid, tanh)
- Training with backpropagation
""",
    "Convolutional Neural Networks": """# Convolutional Neural Networks

CNNs are specialized for image data.  
They use convolutional layers to detect patterns like edges, textures, and shapes.

**Why it matters:**
- Foundation of computer vision
- Used in facial recognition, medical imaging, self-driving cars

**Key ideas:**
- Convolution filters
- Pooling layers (max/avg pooling)
- Fully connected layers for classification
""",
    "Recurrent Neural Networks": """# Recurrent Neural Networks

RNNs are designed for **sequential data** like text, audio, or time series.  
They maintain hidden states to capture past information.

**Why it matters:**
- Great for language modeling, speech recognition, stock prediction

**Key ideas:**
- Loops in architecture
- Vanishing/exploding gradient problems
- Works well for short sequences
""",
    "Long Short-Term Memory": """# LSTMs

LSTMs are a special type of RNN designed to capture long-term dependencies.  
They use gates to control what to keep and forget.

**Why it matters:**
- Handles long sequences better than vanilla RNNs
- Widely used in NLP, translation, speech

**Key ideas:**
- Input, forget, and output gates
- Cell state preserves information
""",
    "Transformers": """# Transformers

Transformers revolutionized NLP by replacing recurrence with attention.  
They focus on **relationships between all words in a sequence**.

**Why it matters:**
- Basis for GPT, BERT, and modern LLMs
- Scales well to large datasets

**Key ideas:**
- Attention mechanism
- Encoder-decoder structure
- Positional embeddings
""",
    "NLP Basics": """# NLP Basics

Natural Language Processing (NLP) allows machines to understand text.  
Preprocessing is crucial before training models.

**Key steps:**
- Tokenization
- Stopword removal
- Stemming and lemmatization

**Why it matters:**
Clean data → better models.
""",
    "Word Embeddings": """# Word Embeddings

Word embeddings map words into vectors in a continuous space.  
Similar words have similar embeddings.

**Why it matters:**
- Captures meaning better than one-hot encoding
- Basis for modern NLP

**Key ideas:**
- Word2Vec, GloVe
- Cosine similarity
""",
    "Generative Adversarial Networks": """# GANs

GANs involve two networks: a **generator** and a **discriminator**.  
They compete in a zero-sum game.

**Why it matters:**
- Generates realistic images, audio, video
- Used in art, super-resolution, data augmentation

**Key ideas:**
- Generator tries to fool the discriminator
- Discriminator distinguishes real vs fake
""",
    "Reinforcement Learning": """# Reinforcement Learning

RL is about training an agent to interact with an environment.  
The agent learns by receiving rewards or penalties.

**Why it matters:**
- Used in robotics, gaming (AlphaGo), resource optimization

**Key ideas:**
- State, action, reward
- Exploration vs exploitation
""",
    "End-to-End ML Project": """# End-to-End ML Project

Now it's time to integrate everything:
- Data preprocessing
- Feature engineering
- Model training
- Evaluation

**Why it matters:**
Real projects require putting all the steps together.
""",
    "AI in Production": """# AI in Production

Building a model is only step one.  
Deploying it reliably is the real challenge.

**Key ideas:**
- Save/load models
- Monitoring
- Scalability

**Why it matters:**
Production readiness makes AI useful in the real world.
""",
    "Mini NLP Project": """# Mini NLP Project

A focused NLP task: e.g., sentiment analysis or text classification.

**Key steps:**
- Data cleaning
- Feature extraction
- Model training & evaluation
""",
    "Capstone Project": """# Capstone Project

The final project!  
Bring together everything you’ve learned into a single end-to-end project.

**Possible ideas:**
- Image classifier
- Text classifier
- Recommendation system
"""
}

# Demo snippets (short, runnable examples)
demo_snippets = {
    "Python Basics": """# Example: simple function
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))""",
    "NumPy & Pandas": """import numpy as np
import pandas as pd

arr = np.array([1, 2, 3, 4, 5])
print("Mean:", arr.mean())

df = pd.DataFrame({"A": [1,2,3], "B": [4,5,6]})
print(df.describe())""",
    "Visualization & Statistics": """import matplotlib.pyplot as plt
import numpy as np

data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30)
plt.title("Histogram")
plt.show()""",
    "Linear Regression": """import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([1.2,1.9,3.2,3.9,5.1])

model = LinearRegression()
model.fit(X,y)
plt.scatter(X,y)
plt.plot(X, model.predict(X), color="red")
plt.show()""",
    "Logistic Regression": """from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y = load_breast_cancer(return_X_y=True)
model = LogisticRegression(max_iter=5000)
model.fit(X,y)
print("Accuracy:", accuracy_score(y, model.predict(X)))""",
    "Decision Trees": """from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X,y = iris.data, iris.target
model = DecisionTreeClassifier(max_depth=3)
model.fit(X,y)

plt.figure(figsize=(10,6))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()""",
    "Random Forests": """from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=50)
model.fit(X,y)
print("Accuracy:", accuracy_score(y, model.predict(X)))""",
    "Support Vector Machines": """from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)
model = SVC(kernel='linear')
model.fit(X,y)
print("Accuracy:", accuracy_score(y, model.predict(X)))""",
    "K-Means Clustering": """from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
model = KMeans(n_clusters=3)
labels = model.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.show()""",
    "Neural Networks": """from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y = load_digits(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = MLPClassifier(hidden_layer_sizes=(32,))
model.fit(X_train,y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))""",
    "Convolutional Neural Networks": """import tensorflow as tf
from tensorflow.keras import layers, models

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1,28,28,1)/255.0
X_test = X_test.reshape(-1,28,28,1)/255.0

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=1,validation_data=(X_test,y_test))""",
    "Recurrent Neural Networks": """import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

X = np.random.random((100,10,1))
y = np.random.randint(2, size=(100,1))

model = models.Sequential([
    layers.SimpleRNN(16, input_shape=(10,1)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,epochs=2)""",
    "Long Short-Term Memory": """import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

X = np.random.random((100,10,1))
y = np.random.randint(2, size=(100,1))

model = models.Sequential([
    layers.LSTM(16, input_shape=(10,1)),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X,y,epochs=2)""",
    "Transformers": """from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Transformers are amazing for NLP!"))""",
    "NLP Basics": """import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

text = "AI is transforming the world."
tokens = word_tokenize(text)
print(tokens)""",
    "Word Embeddings": """import gensim.downloader as api

wv = api.load('glove-wiki-gigaword-50')
print(wv.most_similar('king'))""",
    "Generative Adversarial Networks": """import tensorflow as tf
from tensorflow.keras import layers

# Simple GAN generator
generator = tf.keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(10,)),
    layers.Dense(1, activation="sigmoid")
])
print(generator.summary())""",
    "Reinforcement Learning": """import gym

env = gym.make("CartPole-v1")
obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs = env.reset()
env.close()""",
    "End-to-End ML Project": """from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = RandomForestRegressor()
model.fit(X_train,y_train)
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))""",
    "AI in Production": """from joblib import dump, load
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3]])
y = np.array([2,4,6])

model = LinearRegression().fit(X,y)
dump(model, "linear_model.joblib")
loaded = load("linear_model.joblib")
print(loaded.predict([[4]]))""",
    "Mini NLP Project": """from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

data = fetch_20newsgroups(subset="train")
X_train,y_train = data.data, data.target
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train,y_train)

test = fetch_20newsgroups(subset="test")
print("Accuracy:", accuracy_score(test.target, model.predict(test.data)))""",
    "Capstone Project": """# Placeholder for final project
print("Start building your capstone project here!")"""
}

def create_notebook(topic_id, title, intro, demo):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    # Title
    nb['cells'].append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Intro text
    nb['cells'].append(nbf.v4.new_markdown_cell(intro))

    # Demo code
    nb['cells'].append(nbf.v4.new_code_cell(demo))

    # Exercises
    nb['cells'].append(nbf.v4.new_markdown_cell("## Exercises\n\nTry these on your own:\n- Modify the example above\n- Add a new dataset or parameter\n- Visualize results\n"))

    nb['cells'].append(nbf.v4.new_code_cell("# Your code here\n"))

    return nb

def main():
    os.makedirs("notebooks", exist_ok=True)

    for tid, title, desc in topics:
        intro = intros.get(title, desc)
        demo = demo_snippets.get(title, "# Example code coming soon")
        nb = create_notebook(tid, title, intro, demo)
        with open(f"notebooks/{tid}.ipynb", "w", encoding="utf-8") as f:
            nbf.write(nb, f)

if __name__ == "__main__":
    main()
