import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)

def get_Chat_response(input):
    text_path = "C:/Users/srajp/OneDrive/Desktop/GenAI/ChatBot/split_data/texts.pkl"
    texts1 = load_texts(text_path)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    index_path = "C:/Users/srajp/OneDrive/Desktop/GenAI/ChatBot/Vector_db/faiss_index"
    db1 = load_faiss_index(model_name, index_path)

    question = input
    results = db1.similarity_search_with_relevance_scores(question, k=3)

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    context_list = [(doc, _score) for doc, _score in results]
    prompt = generate_prompt(context_list, question, PROMPT_TEMPLATE)

    token = retrieve_api_token()
    if token:
        cleaned_output = invoke_llm("mistralai/Mistral-7B-Instruct-v0.3", prompt, token)
        return cleaned_output

# Function to load texts from a pickle file
def load_texts(file_path):
    with open(file_path, 'rb') as f:
        texts = pickle.load(f)
    return texts

# Function to load a FAISS index
def load_faiss_index(model_name, index_path):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return db

# Function to generate a prompt using a given template
def generate_prompt(context_list, question, template):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context_list])
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=question)
    return prompt

# Function to retrieve API token from .env file
def retrieve_api_token():
    load_dotenv()
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_token:
        return api_token
    else:
        print("Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
        return None

# Function to invoke an LLM with a given prompt
def invoke_llm(repo_id, prompt, token, temperature=0.1, max_length=512):
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=temperature, model_kwargs={"max_length": max_length, "token": token},add_to_git_credential=True)
    output = llm.invoke(prompt)
    cleaned_output = output.replace("'", "").replace("\n", " ")
    
    return cleaned_output

if __name__ == '__main__':
    app.run()
