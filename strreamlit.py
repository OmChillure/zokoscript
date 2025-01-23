import streamlit as st
import openai
import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = "us-east-1" 
INDEX_NAME = "zoko-chat-index"
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo" 
TOP_K = 140

@st.cache_resource
def init_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    existing = [idx.name for idx in pc.list_indexes()]
    return None if INDEX_NAME not in existing else pc.Index(INDEX_NAME)

def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    return openai.Embedding.create(model=model, input=[text])["data"][0]["embedding"]

def query_and_analyze(query: str, pine_index, model=CHAT_MODEL):
    query_vector = get_embedding(query)
    results = pine_index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    
    if not results.get("matches"):
        return "No relevant matches found in the database."

    context_data = []
    for match in results["matches"]:
        metadata = match["metadata"]
        context_data.append({
            "from_user": metadata.get("from_user"),
            "from_channel": metadata.get("from_channel_id"),
            "to_user": metadata.get("to_user"),
            "to_channel": metadata.get("to_channel_id"),
            "date": metadata.get("date"),
            "message": metadata.get("original_text"),
            "chat_label": metadata.get("chat_label"),
            "score": match.get("score")
        })

    system_prompt = """You are an advanced chat analyst capable of:
    1. Numerical analysis (counting unique numbers, users, message patterns)
    2. Text analysis (message content, sentiment, patterns)
    3. Temporal analysis (dates, frequencies, trends)
    4. Relationship analysis (user interactions, conversation flows)
    
    Extract and analyze exactly what is asked in the query. 
    For counts, provide specific numbers.
    For patterns, provide concrete examples.
    Be precise and data-driven in your analysis."""

    user_prompt = f"""Data Context:
{context_data}

Query: {query}

Analyze the data to answer this query. If counting or identifying patterns, be thorough and exact."""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2 
    )
    
    return response["choices"][0]["message"]["content"].strip()

def main():
    st.title("Advanced Chat Analysis")
    
    pine_index = init_pinecone_index()
    if pine_index is None:
        st.error("Pinecone index not found.")
        return

    st.write("Ask anything about the chats - users, patterns, numbers, messages, dates, etc.")
    
    user_query = st.text_input("Your Query", "")
    if st.button("Analyze"):
        if not user_query.strip():
            st.warning("Please enter a query.")
            return

        with st.spinner("Analyzing..."):
            answer = query_and_analyze(user_query, pine_index)
            st.success("**Analysis:** " + answer)

if __name__ == "__main__":
    main()