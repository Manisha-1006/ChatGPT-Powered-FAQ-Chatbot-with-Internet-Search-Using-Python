from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

# ✅ Correct imports for latest LangChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import SerpAPIWrapper


# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Search tool
search = SerpAPIWrapper(serpapi_api_key=os.getenv("SERPAPI_API_KEY"))
tools = [Tool(name="Google Search", func=search.run, description="Search the web.")]

# Create agent
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("question", "")
    response = agent.run(user_query)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
