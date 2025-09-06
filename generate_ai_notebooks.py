import nbformat as nbf
import os

# ----------------------------
# Define notebook structure
# ----------------------------
roadmap = {
    "week01": [
        ("01_python_basics.ipynb", "Python Basics", [
            {"type": "markdown", "content": "## Exercise 1\nPrint `Hello, AI!`"},
            {"type": "code", "content": "print('Hello, AI!')"},
            {"type": "markdown", "content": "## Exercise 2\nCreate a variable `x = 5` and print it."},
            {"type": "code", "content": "x = 5\nprint(x)"}
        ]),
        ("02_data_types.ipynb", "Data Types & Variables", [
            {"type": "markdown", "content": "## Exercise 1\nCreate an integer, float, string, and boolean variable."},
            {"type": "code", "content": "age = 25\nheight = 5.9\nname = 'Alice'\nis_student = True\nprint(age, height, name, is_student)"}
        ]),
    ],
    "week02": [
        ("01_numpy.ipynb", "NumPy Fundamentals", [
            {"type": "markdown", "content": "## Exercise 1\nCreate a NumPy array of numbers 1–10."},
            {"type": "code", "content": "import numpy as np\narr = np.arange(1, 11)\nprint(arr)"}
        ]),
        ("02_pandas.ipynb", "Pandas for DataFrames", [
            {"type": "markdown", "content": "## Exercise 1\nLoad `customers.csv` into a pandas DataFrame."},
            {"type": "code", "content": "import pandas as pd\ndf = pd.read_csv('../datasets/customers.csv')\ndf.head()"}
        ]),
    ],
    # You can continue expanding for all 22 weeks...
}

# ----------------------------
# Notebook generator
# ----------------------------
def create_notebook(path, title, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    # Add title cell
    nb['cells'].append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Add content cells
    for cell in cells:
        if cell["type"] == "markdown":
            nb['cells'].append(nbf.v4.new_markdown_cell(cell["content"]))
        elif cell["type"] == "code":
            nb['cells'].append(nbf.v4.new_code_cell(cell["content"]))

    # Write notebook file
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

# ----------------------------
# Run the generator
# ----------------------------
if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    # placeholder datasets
    with open("datasets/customers.csv", "w") as f:
        f.write("id,name,age\n1,Alice,30\n2,Bob,25\n")

    with open("datasets/sales.csv", "w") as f:
        f.write("order_id,amount\n1001,250\n1002,400\n")

    with open("datasets/students.csv", "w") as f:
        f.write("id,name,grade\n1,John,A\n2,Sara,B\n")

    for week, notebooks in roadmap.items():
        week_path = os.path.join("notebooks", week)
        os.makedirs(week_path, exist_ok=True)

        for filename, title, cells in notebooks:
            notebook_path = os.path.join(week_path, filename)
            create_notebook(notebook_path, title, cells)

    print("✅ All notebooks generated successfully!")
