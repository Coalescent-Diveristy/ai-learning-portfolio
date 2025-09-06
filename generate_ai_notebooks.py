import nbformat as nbf
import os

# ----------------------------
# Full Roadmap (22 Weeks)
# ----------------------------
roadmap = {
    "week01": [
        ("01_python_basics.ipynb", "Python Basics", [
            {"type": "markdown", "content": "## Exercise 1\nPrint `Hello, AI!`"},
            {"type": "code", "content": "print('Hello, AI!')"},
        ]),
        ("02_data_types.ipynb", "Data Types & Variables", [
            {"type": "markdown", "content": "## Exercise 1\nCreate integer, float, string, and boolean variables."},
            {"type": "code", "content": "age = 25\nheight = 5.9\nname = 'Alice'\nis_student = True\nprint(age, height, name, is_student)"},
        ]),
    ],
    "week02": [
        ("01_numpy.ipynb", "NumPy Fundamentals", [
            {"type": "markdown", "content": "## Exercise 1\nCreate a NumPy array of numbers 1–10."},
            {"type": "code", "content": "import numpy as np\narr = np.arange(1, 11)\nprint(arr)"},
        ]),
        ("02_pandas.ipynb", "Pandas for DataFrames", [
            {"type": "markdown", "content": "## Exercise 1\nLoad `customers.csv` into a pandas DataFrame."},
            {"type": "code", "content": "import pandas as pd\ndf = pd.read_csv('../datasets/customers.csv')\ndf.head()"},
        ]),
    ],
    "week03": [
        ("01_visualization.ipynb", "Data Visualization (Matplotlib)", [
            {"type": "markdown", "content": "## Exercise 1\nPlot a simple line graph of sales."},
            {"type": "code", "content": "import matplotlib.pyplot as plt\nsales = [100,200,300,250]\nplt.plot(sales)\nplt.title('Sales Over Time')\nplt.show()"},
        ]),
        ("02_statistics.ipynb", "Basic Statistics", [
            {"type": "markdown", "content": "## Exercise 1\nCompute mean and median of an array."},
            {"type": "code", "content": "import numpy as np\ndata = [5,10,15,20]\nprint('Mean:', np.mean(data))\nprint('Median:', np.median(data))"},
        ]),
    ],
    "week04": [
        ("01_linear_regression.ipynb", "Linear Regression", [
            {"type": "markdown", "content": "## Exercise 1\nFit a simple linear regression model."},
            {"type": "code", "content": "from sklearn.linear_model import LinearRegression\nimport numpy as np\nX = np.array([[1],[2],[3],[4]])\ny = np.array([2,4,6,8])\nmodel = LinearRegression().fit(X,y)\nprint(model.coef_, model.intercept_)"},
        ]),
    ],
    "week05": [
        ("01_logistic_regression.ipynb", "Logistic Regression", [
            {"type": "markdown", "content": "## Exercise 1\nTrain a logistic regression classifier."},
            {"type": "code", "content": "from sklearn.linear_model import LogisticRegression\nimport numpy as np\nX = np.array([[0],[1],[2],[3]])\ny = np.array([0,0,1,1])\nclf = LogisticRegression().fit(X,y)\nprint(clf.predict([[1.5],[2.5]]))"},
        ]),
    ],
    "week06": [
        ("01_decision_trees.ipynb", "Decision Trees", [
            {"type": "markdown", "content": "## Exercise 1\nTrain a decision tree classifier."},
            {"type": "code", "content": "from sklearn.tree import DecisionTreeClassifier\nimport numpy as np\nX = np.array([[0],[1],[2],[3]])\ny = np.array([0,0,1,1])\nclf = DecisionTreeClassifier().fit(X,y)\nprint(clf.predict([[1.5]]))"},
        ]),
    ],
    "week07": [
        ("01_random_forests.ipynb", "Random Forests", [
            {"type": "markdown", "content": "## Exercise 1\nTrain a random forest classifier."},
            {"type": "code", "content": "from sklearn.ensemble import RandomForestClassifier\nimport numpy as np\nX = np.array([[0],[1],[2],[3]])\ny = np.array([0,0,1,1])\nclf = RandomForestClassifier().fit(X,y)\nprint(clf.predict([[2]]))"},
        ]),
    ],
    "week08": [
        ("01_svm.ipynb", "Support Vector Machines", [
            {"type": "markdown", "content": "## Exercise 1\nTrain a simple SVM classifier."},
            {"type": "code", "content": "from sklearn.svm import SVC\nimport numpy as np\nX = np.array([[0,0],[1,1],[1,0],[0,1]])\ny = [0,1,0,1]\nclf = SVC().fit(X,y)\nprint(clf.predict([[0.9,0.9]]))"},
        ]),
    ],
    "week09": [
        ("01_kmeans.ipynb", "Clustering (K-Means)", [
            {"type": "markdown", "content": "## Exercise 1\nCluster customers by age and spending score."},
            {"type": "code", "content": "from sklearn.cluster import KMeans\nimport numpy as np\nX = np.array([[20,200],[30,400],[25,350],[40,800]])\nkmeans = KMeans(n_clusters=2, random_state=42).fit(X)\nprint(kmeans.labels_)"},
        ]),
    ],
    "week10": [
        ("01_intro_neural_networks.ipynb", "Neural Networks", [
            {"type": "markdown", "content": "## Exercise 1\nBuild a simple feedforward NN in Keras."},
            {"type": "code", "content": "from tensorflow import keras\nfrom tensorflow.keras import layers\nmodel = keras.Sequential([\n    layers.Dense(8, activation='relu', input_shape=(4,)),\n    layers.Dense(1, activation='sigmoid')\n])\nmodel.compile(optimizer='adam', loss='binary_crossentropy')\nmodel.summary()"},
        ]),
    ],
    "week11": [
        ("01_cnn.ipynb", "Convolutional Neural Networks", [
            {"type": "markdown", "content": "## Exercise 1\nBuild a simple CNN for image data."},
            {"type": "code", "content": "from tensorflow.keras import layers, models\nmodel = models.Sequential([\n    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),\n    layers.MaxPooling2D((2,2)),\n    layers.Flatten(),\n    layers.Dense(10,activation='softmax')\n])\nmodel.summary()"},
        ]),
    ],
    "week12": [
        ("01_rnn.ipynb", "Recurrent Neural Networks", [
            {"type": "markdown", "content": "## Exercise 1\nBuild a simple RNN for sequence prediction."},
            {"type": "code", "content": "from tensorflow.keras import layers, models\nmodel = models.Sequential([\n    layers.SimpleRNN(32,input_shape=(10,1)),\n    layers.Dense(1)\n])\nmodel.summary()"},
        ]),
    ],
    "week13": [
        ("01_lstm.ipynb", "LSTMs", [
            {"type": "markdown", "content": "## Exercise 1\nBuild a simple LSTM model."},
            {"type": "code", "content": "from tensorflow.keras import layers, models\nmodel = models.Sequential([\n    layers.LSTM(32,input_shape=(10,1)),\n    layers.Dense(1)\n])\nmodel.summary()"},
        ]),
    ],
    "week14": [
        ("01_transformers.ipynb", "Transformers", [
            {"type": "markdown", "content": "## Exercise 1\nLoad a pretrained transformer for text classification."},
            {"type": "code", "content": "from transformers import pipeline\nclassifier = pipeline('sentiment-analysis')\nprint(classifier('AI is amazing!'))"},
        ]),
    ],
    "week15": [
        ("01_nlp_basics.ipynb", "Natural Language Processing Basics", [
            {"type": "markdown", "content": "## Exercise 1\nTokenize a text using NLTK."},
            {"type": "code", "content": "import nltk\nnltk.download('punkt')\nfrom nltk.tokenize import word_tokenize\nprint(word_tokenize('Artificial Intelligence is fascinating.'))"},
        ]),
    ],
    "week16": [
        ("01_embeddings.ipynb", "Word Embeddings", [
            {"type": "markdown", "content": "## Exercise 1\nLoad GloVe embeddings and find word similarity."},
            {"type": "code", "content": "# Example with spaCy\nimport spacy\nnlp = spacy.load('en_core_web_md')\nprint(nlp('king').similarity(nlp('queen')))"},
        ]),
    ],
    "week17": [
        ("01_gans.ipynb", "Generative Adversarial Networks", [
            {"type": "markdown", "content": "## Exercise 1\nBuild a simple GAN structure (no training)."},
            {"type": "code", "content": "from tensorflow.keras import layers\nfrom tensorflow.keras.models import Sequential\n# Generator\ngenerator = Sequential([\n    layers.Dense(16, activation='relu', input_shape=(10,)),\n    layers.Dense(32, activation='relu'),\n    layers.Dense(64, activation='sigmoid')\n])\ngenerator.summary()"},
        ]),
    ],
    "week18": [
        ("01_reinforcement_learning.ipynb", "Reinforcement Learning Basics", [
            {"type": "markdown", "content": "## Exercise 1\nRun a simple OpenAI Gym environment."},
            {"type": "code", "content": "import gym\nenv = gym.make('CartPole-v1')\nobs = env.reset()\nfor _ in range(5):\n    action = env.action_space.sample()\n    obs, reward, done, info, _ = env.step(action)\n    if done:\n        obs = env.reset()\nprint('Sample run complete.')"},
        ]),
    ],
    "week19": [
        ("01_ml_project.ipynb", "End-to-End ML Project", [
            {"type": "markdown", "content": "## Exercise 1\nLoad a dataset, preprocess it, and train a model."},
            {"type": "code", "content": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n\n# Load\nurl = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'\ndf = pd.read_csv(url)\n\n# Prep\nX = df.drop('species',axis=1)\ny = df['species']\nX_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)\n\n# Train\nclf = LogisticRegression(max_iter=200).fit(X_train,y_train)\nprint('Accuracy:', clf.score(X_test,y_test))"},
        ]),
    ],
    "week20": [
        ("01_ai_in_production.ipynb", "Deploying AI Models", [
            {"type": "markdown", "content": "## Exercise 1\nSave and load a scikit-learn model with joblib."},
            {"type": "code", "content": "from sklearn.linear_model import LogisticRegression\nimport joblib\nimport numpy as np\n\nX = np.array([[1],[2],[3]])\ny = [0,1,1]\nclf = LogisticRegression().fit(X,y)\n\njoblib.dump(clf, 'model.pkl')\nloaded = joblib.load('model.pkl')\nprint(loaded.predict([[2]]))"},
        ]),
    ],
    "week21": [
        ("01_mini_project_1.ipynb", "Mini Project 1 - NLP Sentiment Analysis", [
            {"type": "markdown", "content": "## Task\nBuild a sentiment analysis classifier using IMDb dataset."},
        ]),
    ],
    "week22": [
        ("01_capstone_project.ipynb", "Capstone Project - End-to-End AI", [
            {"type": "markdown", "content": "## Task\nChoose a dataset and build a full pipeline: data cleaning → model training → evaluation → deployment-ready artifact."},
        ]),
    ],
}

# ----------------------------
# Notebook generator
# ----------------------------
def create_notebook(path, title, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []
    nb['cells'].append(nbf.v4.new_markdown_cell(f"# {title}"))
    for cell in cells:
        if cell["type"] == "markdown":
            nb['cells'].append(nbf.v4.new_markdown_cell(cell["content"]))
        elif cell["type"] == "code":
            nb['cells'].append(nbf.v4.new_code_cell(cell["content"]))
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

# ----------------------------
# Run generator
# ----------------------------
if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    # Sample datasets
    with open("datasets/customers.csv", "w") as f:
        f.write("id,name,age\n1,Alice,30\n2,Bob,25\n")
    with open("datasets/sales.csv", "w") as f:
        f.write("order_id,amount\n1001,250\n1002,400\n")
    with open("datasets/students.csv", "w") as f:
        f.write("id,name,grade\n1,John,A\n2,Sara,B\n")

    # Generate notebooks
    for week, notebooks in roadmap.items():
        week_path = os.path.join("notebooks", week)
        os.makedirs(week_path, exist_ok=True)
        for filename, title, cells in notebooks:
            notebook_path = os.path.join(week_path, filename)
            create_notebook(notebook_path, title, cells)

    print("✅ All 22 weeks of notebooks generated successfully!")
