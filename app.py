from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize Flask app
app = Flask(__name__)

# Sample documents (replace with your own data)
documents = [
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "Guido van Rossum is the creator of Python, which was first released in 1991.",
    "Python's design philosophy emphasizes code readability with its notable use of significant indentation.",
    "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
    "The Python community is known for being welcoming and inclusive, contributing to its widespread adoption.",
    "Python's extensive standard library is often referred to as 'batteries included' due to its comprehensive functionality.",
    "Python is widely used in data science, machine learning, web development, and automation due to its versatility.",
    "The Python Package Index (PyPI) hosts thousands of third-party libraries, making it easy to extend Python's capabilities.",
    "Python's syntax is designed to be intuitive, making it an excellent choice for beginners learning to code.",
    "The Zen of Python, written by Tim Peters, outlines the guiding principles for writing Pythonic code.",
    "Python's dynamic typing and automatic memory management make it a flexible and developer-friendly language.",
    "Python 2 and Python 3 are two major versions of the language, with Python 3 being the current and recommended version.",
    "Python's popularity has grown significantly over the years, consistently ranking as one of the top programming languages.",
    "Frameworks like Django and Flask have made Python a popular choice for web development.",
    "Python's integration with scientific libraries like NumPy, pandas, and Matplotlib has made it a favorite in the data science community.",
    "The Global Interpreter Lock (GIL) in Python can be a limitation for CPU-bound multithreaded programs.",
    "Python's simplicity and readability often lead to faster development cycles compared to other programming languages.",
    "Python's ecosystem includes powerful tools like Jupyter Notebooks, which are widely used for interactive computing.",
    "Python's compatibility with major operating systems (Windows, macOS, Linux) contributes to its cross-platform appeal.",
    "Python's emphasis on community-driven development has led to a rich ecosystem of tutorials, forums, and resources for learners."
]

# Step 1: Embed documents using a retriever model
retriever = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
document_embeddings = retriever.encode(documents, convert_to_tensor=True).cpu().numpy()

# Step 2: Build a FAISS index for efficient search
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Step 3: Define a retriever function
def retrieve_documents(query, k=2):
    query_embedding = retriever.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# Step 4: Load a generator model (e.g., GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
generator = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 5: Generate an answer using retrieved context
def generate_answer(query):
    # Retrieve relevant documents
    context_docs = retrieve_documents(query)
    context = " ".join(context_docs)
    
    # Create prompt with context and query
    prompt = f"Answer the question based on the context below.\nContext: {context}\nQuestion: {query}\nAnswer:"
    
    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = generator.generate(
        inputs.input_ids,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle queries
@app.route('/ask', methods=['POST'])
def ask():
    # Get the query from the request
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Generate the answer
    answer = generate_answer(query)
    
    # Return the answer as JSON
    return jsonify({"query": query, "answer": answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)