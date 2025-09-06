# Script to automatically generate all AI learning notebooks with starter code and exercises, including Week 1 exercises
import os

# Define repo structure
notebook_structure = {
    '01_foundations': [
        '01_python_basics',
        '02_pandas_numpy',
        '03_visualization',
        '04_intro_ml'
    ],
    '02_core_ml': [
        '01_linear_regression',
        '02_logistic_regression',
        '03_decision_trees_random_forests',
        '04_clustering',
        '05_feature_engineering',
        '06_mini_projects'
    ],
    '03_deep_learning': [
        '01_neural_networks',
        '02_cnns',
        '03_rnns_lstm',
        '04_pytorch_tensorflow',
        '05_extended_dl_practice'
    ],
    '04_generative_ai': [
        '01_transformers_basics',
        '02_text_generation_summarization',
        '03_chatbot_qa',
        '04_speech_to_text_whisper'
    ],
    '05_capstone': [
        '01_deployment_basics',
        '02_capstone_integration',
        '03_capstone_polishing'
    ]
}

base_path = 'notebooks'
os.makedirs(base_path, exist_ok=True)

# Week 1 starter code with exercises
week1_code = """"""
# 01_python_basics.ipynb

# Week 1: Python Basics

# Variables
integer_var = 10
float_var = 3.14
string_var = 'Hello AI'
boolean_var = True
print(type(integer_var), type(float_var), type(string_var), type(boolean_var))

# List and loop
numbers = [1, 2, 3, 4, 5]
for n in numbers:
    print(n**2)

# Function
s = 'Matthew'
def greet(name):
    return f'Hello, {name}!'
print(greet(s))

# Exercises
# 1. Create a list of your favorite AI topics and print each with a for loop
# 2. Write a function that takes a number and returns its factorial
# 3. Create a dictionary of 3 key-value pairs and print each key and value
""""""

# Generic starter code for other notebooks
generic_code = """"""
# {notebook_name}

# This notebook contains starter code and exercises for {topic}

# EXAMPLES
print('Hello AI!')

# EXERCISES
# 1. Exercise 1 description
# 2. Exercise 2 description
""""""

# Create directories and notebooks
for folder, notebooks in notebook_structure.items():
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    for nb in notebooks:
        nb_path = os.path.join(folder_path, nb + '.ipynb')
        if not os.path.exists(nb_path):
            with open(nb_path, 'w') as f:
                if folder == '01_foundations' and nb == '01_python_basics':
                    f.write(week1_code)
                else:
                    f.write(generic_code.replace('{notebook_name}', nb).replace('{topic}', folder))

print('All starter notebooks including Week 1 exercises generated successfully.')
