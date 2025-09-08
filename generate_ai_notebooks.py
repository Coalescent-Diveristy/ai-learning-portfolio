import nbformat as nbf
import os

# Full roadmap (26 topics including LLMs)
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
    ("15_llm_basics", "LLM Basics", "Understand how large language models are built and used."),
    ("16_pretrained_llms", "Using Pretrained LLMs", "Leverage Hugging Face and APIs to use pretrained language models."),
    ("17_llm_apps", "LLM-Powered Applications", "Build AI apps like chatbots, summarizers, and assistants."),
    ("18_advanced_llms", "Advanced LLM Techniques", "Explore prompt engineering, fine-tuning, and retrieval-augmented generation."),
    ("19_nlp_basics", "NLP Basics", "Preprocess and analyze text data."),
    ("20_embeddings", "Word Embeddings", "Learn about word2vec, GloVe, and embedding spaces."),
    ("21_gans", "Generative Adversarial Networks", "Understand GANs and how they generate data."),
    ("22_rl", "Reinforcement Learning", "Explore agents, environments, and rewards."),
    ("23_end_to_end", "End-to-End ML Project", "Combine skills into a small project."),
    ("24_ai_production", "AI in Production", "Learn model saving, loading, and deployment basics."),
    ("25_project_nlp", "Mini NLP Project", "Build and evaluate an NLP pipeline."),
    ("26_capstone", "Capstone Project", "Do a full end-to-end AI project."),
]

# Detailed intros for new LLM topics
intros = {
    "LLM Basics": """# LLM Basics

Large Language Models (LLMs) are deep learning models trained on vast text corpora to understand and generate human-like text.

**Key ideas:**
- Tokenization: breaking text into tokens.
- Pretraining: predicting the next word over billions of examples.
- Fine-tuning: adapting to specific tasks.
- Inference: generating answers to prompts.

**Why it matters:**
LLMs power tools like ChatGPT, Copilot, and other modern AI assistants.
""",
    "Using Pretrained LLMs": """# Using Pretrained LLMs

Instead of training models from scratch, we can use pretrained LLMs from libraries like Hugging Face or APIs like OpenAI.

**Benefits:**
- Saves compute and time.
- Access to state-of-the-art models.
- Easy integration with Python code.

**Why it matters:**
This is the most practical way to build LLM-powered applications.
""",
    "LLM-Powered Applications": """# LLM-Powered Applications

Once you can access an LLM, you can integrate it into real applications.

**Examples:**
- Chatbots
- Q&A systems
- Summarizers
- Assistants for code, writing, or research

**Why it matters:**
This is where AI becomes interactive and directly useful.
""",
    "Advanced LLM Techniques": """# Advanced LLM Techniques

To make LLMs truly powerful, we use advanced techniques.

**Topics:**
- Prompt engineering
- Fine-tuning (adapting models to your data)
- Retrieval-Augmented Generation (RAG)
- Using embeddings and vector databases

**Why it matters:**
These techniques let you customize and scale LLMs for production.
"""
}

# Demo snippets for LLM notebooks
demo_snippets = {
    "LLM Basics": """from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time,", max_length=30))""",
    "Using Pretrained LLMs": """from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
print(sentiment("I love learning AI with Jupyter notebooks!"))""",
    "LLM-Powered Applications": """from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

# Make sure you set your API key:
# os.environ["OPENAI_API_KEY"] = "your-key"

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(input_variables=["question"], template="Answer as an AI tutor: {question}")

print(llm(prompt.format(question="What is overfitting in machine learning?")))""",
    "Advanced LLM Techniques": """from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

embeddings = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="db", embedding_function=embeddings)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever()
)

print(qa.run("Explain reinforcement learning in simple terms"))"""
}

# Fallback intro if not in dictionary
def get_intro(title, desc):
    if title in intros:
        return intros[title]
    return f"# {title}\n\n{desc}\n\n**Why it matters:**\nUnderstanding {title.lower()} is essential for building AI systems."

# Fallback demo if not in dictionary
def get_demo(title):
    if title in demo_snippets:
        return demo_snippets[title]
    return "# Demo code example for " + title

# Exercise template
def get_exercise(title):
    return f"## Exercise: {title} Practice\n\n- Apply {title} to a dataset or task.\n- Experiment with parameters.\n- Visualize or log results.\n- Reflect: What did you learn?\n"

def create_notebook(folder, title, desc):
    nb = nbf.v4.new_notebook()
    cells = []

    # Intro
    cells.append(nbf.v4.new_markdown_cell(get_intro(title, desc)))

    # Demo
    cells.append(nbf.v4.new_code_cell(get_demo(title)))

    # Exercise
    cells.append(nbf.v4.new_markdown_cell(get_exercise(title)))
    cells.append(nbf.v4.new_code_cell("# Your code here\n"))

    nb['cells'] = cells
    return nb

def main():
    os.makedirs("notebooks", exist_ok=True)
    for folder, title, desc in topics:
        nb = create_notebook(folder, title, desc)
        path = os.path.join("notebooks", f"{folder}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
    print("✅ All 26 notebooks generated with intros, demos, and exercises.")

if __name__ == "__main__":
    main()
