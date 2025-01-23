import streamlit as st
import openai
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = "us-east-1" 
INDEX_NAME = "zoko-chat-index"
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"          
TOP_K = 3                           

@st.cache_resource
def init_pinecone_index():
    """
    Create/connect to the Pinecone index. 
    We assume it's already populated with your embedded messages.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        st.error(f"Pinecone index '{INDEX_NAME}' does not exist. "
                 f"Create or populate it first.")
        return None

    return pc.Index(INDEX_NAME)

# ---------------------------
# 3) HELPER FUNCTIONS
# ---------------------------

def get_embedding(text: str, model: str = EMBED_MODEL) -> list:
    """Create an embedding vector for the given text using OpenAI."""
    resp = openai.Embedding.create(
        model=model,
        input=[text]
    )
    return resp["data"][0]["embedding"]

def semantic_search(query: str, pine_index, top_k: int = TOP_K):
    """
    1) Embeds 'query' with OpenAI
    2) Searches Pinecone
    3) Returns the raw matches (with metadata)
    """
    query_vector = get_embedding(query)
    result = pine_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return result

def llm_analysis(context_texts: list, question: str, model=CHAT_MODEL) -> str:
    """
    Feed top contexts + question to GPT and return the response.
    """
    context_str = "\n".join(context_texts)

    system_prompt = "You are a helpful assistant that interprets these user messages and provides a detailed answer."

    user_prompt = f"""Here are some chat messages from our data:

{context_str}

Question: {question}

Please provide a thorough, detailed, but concise answer or summary.
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"].strip()

def main():
    st.title("Zoko Chat LLM Analysis")

    pine_index = init_pinecone_index()
    if pine_index is None:
        return

    st.write("Enter a query or question about the stored WhatsApp chats. "
             "We will perform semantic search on the embedded messages, "
             "then pass the top matches to GPT for a detailed answer.")

    user_query = st.text_input("Your Query or Prompt", "")
    if st.button("Search & Analyze"):
        if not user_query.strip():
            st.warning("Please enter a query first.")
            return

        st.info(f"Semantic searching for: '{user_query}' ...")
        results = semantic_search(user_query, pine_index, top_k=TOP_K)

        if "matches" not in results or not results["matches"]:
            st.warning("No relevant matches found in Pinecone.")
            return

        context_texts = []
        for match in results["matches"]:
            meta = match["metadata"]
            from_user = meta.get("from_user", "")
            to_user   = meta.get("to_user", "")
            date      = meta.get("date", "")
            original  = meta.get("original_text", "")
            score     = match.get("score", 0)

            st.write(f"**Match ID**: {match['id']}, **Score**: {score:.4f}")
            st.write(f"From: {from_user}  |  To: {to_user}  |  Date: {date}")
            st.write(f"Message: {original}")
            st.write("---")

            # Collect for LLM context
            context_texts.append(
                f"From: {from_user}\nTo: {to_user}\nDate: {date}\nMessage: {original}"
            )

        st.info("Analyzing with GPT...")
        answer = llm_analysis(context_texts, user_query)
        st.success("**Answer:** " + answer)

if __name__ == "__main__":
    main()
