from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers, huggingface_pipeline
from langchain.chains import RetrievalQA
import chainlit as cl
import spacy
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import redis

device = "cuda" if torch.cuda.is_available() else "cpu"

# Redis for response caching
cache = redis.StrictRedis(host='localhost', port=6379, db=0)

model_name = "colbert-ir/colbertv2.0"
colbert_model = SentenceTransformer(model_name)
colbert_tokenizer = AutoTokenizer.from_pretrained(model_name)

DB_FAISS_PATH = "vectorstore\db_faiss_ML"
acclerator = Accelerator()
nlp = spacy.load("en_core_web_sm")

# Custom prompt template
custom_prompt_template = """ScidentAI is committed to providing exceptional service in a respectful and truthful manner. My responses will be clear, concise, and free of bias, fostering a positive and inclusive environment. I'll always complete my sentences and use meaningful language to answer your questions directly.

**Important Note:** I can't share any confidential information about ScidentAI's internal workings or systems. However, I can leverage my knowledge to assist you with various tasks and answer your questions in an informative way.
**Context:** {context}
**Question:** {question}
**Important Note:** If there is nothing available in Context then simply and politely say that i dont know the answer but donot produce unwanted answers.
Here's my best shot at a helpful and informative answer:
Helpful answer:
"""

# Text extraction and caching
def extract_and_cache_text(pdf_path):
    cache_path = pdf_path + ".txt"
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as file:
            return file.read()
    else:
        loader = PyPDFLoader(pdf_path)
        text = loader.load_and_cache_text()  # Assume this method caches the text
        with open(cache_path, 'w') as file:
            file.write(text)
        return text

# Batch processing for embedding generation
def batch_generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    model = SentenceTransformer(model_name, device=acclerator.device)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        embeddings.extend(batch_embeddings)
    return embeddings

# Colbert Reranking
def colbert_reranking(query_embedding, retrieved_documents, doc_embeddings):
    colbert_scores = []
    with ThreadPoolExecutor() as executor:
        colbert_scores = list(executor.map(
            lambda doc_embedding: torch.cosine_similarity(query_embedding, doc_embedding).item(),
            doc_embeddings
        ))
    sorted_list = sorted(zip(colbert_scores, retrieved_documents), reverse=True)
    reranked_documents = [doc for _, doc in sorted_list]
    return reranked_documents

# Load FAISS index
def load_faiss_index(embeddings, db_path):
    if os.path.exists(db_path):
        return FAISS.load_local(db_path, device=acclerator.device)
    else:
        index = FAISS.from_documents(embeddings)
        index.save_local(db_path)
        return index

# Set custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

# Loading the LLM
def load_llm():
    config = {"max_new_tokens": 512, "repetition_penalty": 1.1, "context_length": 8000}
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGUF",
        model_type="llama",
        temperature=0.1,
        device="cuda",
        config=config,
    )
    return llm

# QA Bot Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Check if response contains internal info
def is_internal_info(text):
    internal_info_keywords = [
        "internal",
        "confidential",
        "private",
        "implementation",
        "internal workings",
        "SciExtract architecture",
        "Salary of employee",
        "Company ongoing projects",
    ]
    for keyword in internal_info_keywords:
        if keyword.lower() in text.lower():
            return True
    return False

# Update prompt with few-shot examples
def update_prompt_fewshot(conversation_entities, query):
    prompt_prefix = f"Here are some previous examples related to your question:\n"
    examples = []
    for entity, label in conversation_entities.items():
        examples.append(f"- {entity} ({label})")
    prompt_Examples = "\n".join(examples)
    updated_prompt = f"{prompt_prefix}{prompt_Examples}\n **Question:** {query}"
    return updated_prompt

# Retrieve additional context
def retrieve_additional_context(conversation_entities, vectorstore):
    additional_context = []
    for entity, label in conversation_entities.items():
        relevant_info = vectorstore.get_relevant_info(entity, label)
        additional_context.append(relevant_info)
    return "\n".join(additional_context)

# Cache response
def cache_response(query, response):
    cache.set(query, response)

# Get cached response
def get_cached_response(query):
    return cache.get(query)

# Final result function
async def final_result(query):
    try:
        cached_response = get_cached_response(query)
        if cached_response:
            return cached_response.decode('utf-8')

        qa = qa_bot()
        response = await qa.acall({"query": query})

        if is_internal_info(response['result']):
            return "I'm sorry, I can't share information about company or bot internals. However, I'd be happy to answer a different question for you!"

        # Cache the response
        cache_response(query, response['result'])
        return response['result']

    except Exception as e:
        return f"Oops! I encountered an issue while processing your request. (Error: {str(e)}). Please try again later."

# Chainlit handlers
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = """**Welcome to SciExtract!**
                I'm your AI-powered research assistant, ready to help you with Scientific Documents related to Material Science with AI. Ask me anything!"""
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    await cl.Message(content=answer).send()
