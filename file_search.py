import streamlit as st
from txtai import Embeddings
import json
import os
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from uuid import uuid4
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time

st.set_page_config(
    page_title="Lervis Enterprise",
    layout="wide",
)

st.markdown(
    """
    <style>
    .message-container {
        margin-bottom: 20px;
        width: 100%;
    }
    .message-container .user-message {
        text-align: right;
        padding: 10px;
        border-radius: 5px;
        background-color: #0c0912;
        margin-bottom: 20px;
    }
    .message-container .assistant-message {
        text-align: left;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .st-emotion-cache-qdbtli {
        width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


openai = st.secrets.db_credentials.openai 
qdrant = st.secrets.db_credentials.qdrant
qdrant_url = st.secrets.db_credentials.qdrant_url

from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm_nvidia = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5, api_key=openai)


file_path = r"document_splits.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

embeddings = Embeddings(
    content=True,
    defaults=False,
    indexes={
        "keyword": {
            "keyword": True
        },
        "dense": {
            "path": "sentence-transformers/nli-mpnet-base-v2"
        }
    }
)

index_path = "index"

# Check if index directory exists, attempt to load if it does; otherwise, create and save it.
if os.path.exists(index_path):
    try:
        print("Loading existing index...")
        embeddings.load(index_path)
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Error loading index: {e}")
        print("Re-indexing data.")
        embeddings.index(data)
        embeddings.save(index_path)
        print("Index created and saved successfully.")
else:
    print("Index not found. Creating a new index.")
    embeddings.index(data)
    embeddings.save(index_path)
    print("Index created and saved successfully.")

if "first_level_results" not in st.session_state:
    st.session_state.first_level_results = None
if "result_ids" not in st.session_state:
    st.session_state.result_ids = {}
if "second_level_results" not in st.session_state:
    st.session_state.second_level_results = None
if "documents" not in st.session_state:
    st.session_state.documents = []

col1, col2 = st.columns(2)

with col1:
    st.title("File Search")
    query = st.text_area("Enter your search query (multiple keywords separated by commas):", "")

    k_first = st.number_input("Enter the number of results for the first search (k):", min_value=1, max_value=15, value=10)
    k_second = st.number_input("Enter the number of results for the second search (k):", min_value=1, max_value=k_first, value=5)

    if st.button("Search"):
        if query:

            keywords = [keyword.strip() for keyword in query.split(",")]
            
            first_level_results = []
            result_ids_dict = {}
            st.session_state.documents = []

            for keyword in keywords:
                st.write(f"**Query:** {keyword}")
                
                keyword_results = embeddings.search(keyword, k_first, index="keyword")
                
                if keyword_results:
                    st.write(f"**First Level Search Results (k={k_first}):**")
                    for result in keyword_results:
                        with st.container(border=True):
                            st.write(f"**Document:** {result['text']}")
                            st.write(f"**Score:** {result['score']}")
                    sorted_results = sorted(keyword_results, key=lambda x: x['score'], reverse=True)[:10]
                    st.session_state.documents.extend(sorted_results)

                    first_level_results.extend(keyword_results)
                    
                    result_ids_dict[keyword] = [result['id'] for result in keyword_results]
                else:
                    st.write("No results found for this keyword.")
            
            st.session_state.first_level_results = first_level_results
            st.session_state.result_ids = result_ids_dict

        else:
            st.write("Please enter a query to search.")


    if "first_level_results" in st.session_state and st.session_state.first_level_results:
        if st.button("Perform internal Search"):
            second_level_results = []
            st.session_state.documents = []

            for keyword, ids in st.session_state.result_ids.items():
                st.write(f"**Second-level search for: {keyword}**")
                
                keyword_results = [result for result in st.session_state.first_level_results if result['id'] in ids]

                second_level_query = " ".join([result['text'] for result in keyword_results])

                second_level_keyword_results = embeddings.search(second_level_query, k_second, index="keyword")


                if second_level_keyword_results:
                    for result in second_level_keyword_results:
                        with st.container(border=True):
                            st.write(f"**Document:** {result['text']}")
                            st.write(f"**Score:** {result['score']}")

                    sorted_results = sorted(second_level_keyword_results, key=lambda x: x['score'], reverse=True)[:10]
                    st.session_state.documents.extend(sorted_results)

                else:
                    st.write(f"No results found for second-level search on document '{ids}'.")


from langchain.embeddings.base import Embeddings

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [self.model.encode(d,batch_size=64).tolist() for d in documents]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

with col2:
    st.title("RAG for Retrived documents")
    user_input = st.chat_input("Enter your query:",args=(True,))

    if user_input:
        st.markdown(f'<div class="message-container"><p class="user-message">{user_input}</p></div>', unsafe_allow_html=True)

        qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant,
        )
        collection_name = "easa"
        from qdrant_client.http.models import Distance, VectorParams

        try:
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"Error deleting collection: {e}")
    

        if collection_name not in qdrant_client.get_collections():
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

        documents = [Document(page_content=doc['text'], metadata={'id': doc['id']}) for doc in st.session_state.documents]


        qdrant_store = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding_model)
        uuids = [str(uuid4()) for _ in range(len(documents))]

        qdrant_store.add_documents(documents=documents, ids=uuids)
        retriever = qdrant_store.as_retriever()

        #qa_chain = RetrievalQA.from_llm(llm=llm_nvidia, retriever=retriever, chain_type="stuff")
        system_prompt = (
            "You are an expert assistant for answering questions about the "
            "Easy Access Rules for Continuing Airworthiness (Regulation (EU) No 1321/2014). "
            "Use the retrieved context below to provide precise and concise answers. "
            "If the answer is not found in the context, respond by stating 'I don't know.'\n\n"
            "'\n' this means new line."

            "Answer Format:\n"
            "1. RAG Response : [Your concise answer]\n"
            "2. Supporting Content: [Piece of context leading to the answer]\n"
            "3. Document Page: [Page number where information was found]\n\n"
            "Each element should begin on a new line."
            "Use proper formatting with Headers and points"
            
            "Use proper formatting.\n\n"
            "'\n' this means new line. '**' this means bold formatting"
            
            "Retrieved Context:\n{context}"
        )


        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )


        question_answer_chain = create_stuff_documents_chain(llm_nvidia, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        #result = qa_chain.run(user_input)
        result = rag_chain.invoke({"input": user_input})
        st.write("### Agent Response:")
        def stream_data():
            for word in result['answer'].split(" "):
                yield word + " "
                time.sleep(0.02)
        st.write_stream(stream_data)

        #st.markdown(f'<div class="message-container"><p class="assistant-message">{result['answer']}</p></div>', unsafe_allow_html=True)