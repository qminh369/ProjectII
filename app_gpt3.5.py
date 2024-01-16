import gradio as gr
import numpy as np
import string
import os
import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain.vectorstores import FAISS

import openai

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

with open('/home/skmlab/data/quangminh/ProjectII/api_key.txt', 'r') as file:
    openai.api_key = file.readline()

db = pd.read_csv('/home/skmlab/data/quangminh/ProjectII/db.csv')

#model = "gpt-4-1106-preview"
#model = "gpt-3.5-turbo"
model = "gpt-3.5-turbo-1106"

# List Ref
refs = []
for ref in db['ref']:
    refs.append(ref)

# List chunks
chunks = []
for chunk in db['chunk']:
    chunks.append(chunk)

# Initializing the Bi-Encoder model
bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
top_k = 100  # Number of passages to retrieve with the bi-encoder

corpus_embeddings = bi_encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

# Cross encoder (reranking)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')

# Tokenizing the corpus for BM25
tokenized_corpus = [chunk.split(" ") for chunk in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Define a function for search
def search(query):
    # Lexical Search (BM25)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = np.argpartition(bm25_scores, -3)[-3:]
    #bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = [{'corpus_id': idx, 'ref': refs[idx], 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    #top_lexical = [chunks[hit['corpus_id']].replace("\n", " ") for hit in bm25_hits[0:5]]
    ref_lexical = [refs[hit['corpus_id']] for hit in bm25_hits[0:3]]
    bm25_relevant_chunk = []
    for ref in ref_lexical:
        chunk_relevant_query = db['chunk'][db['ref'] == ref].tolist()
        bm25_relevant_chunk.append(chunk_relevant_query)
    
    bm25_relevant_chunk = [item for chunk in bm25_relevant_chunk for item in chunk]
    bm25_relevant_chunk = list(dict.fromkeys(bm25_relevant_chunk))
    
    # Semantic Search
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # Re-Ranking with Cross-Encoder
    cross_inp = [[query, chunks[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Combine scores and present the results
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    new_hits = [{"ref": refs[hit['corpus_id']]} for hit in hits]

    #top_cross_encoder_hits = [chunks[hit['corpus_id']].replace("\n", " ") for hit in hits[0:5]]
    ref_cross_encoder = [refs[hit['corpus_id']] for hit in hits[0:3]]
    
    #result = top_lexical + top_cross_encoder_hits
    #ref_result = ref_lexical + ref_cross_encoder
    
    cross_encoder_relevant_chunk = []
    for ref in ref_cross_encoder:
        chunk_relevant_query = db['chunk'][db['ref'] == ref].tolist()
        cross_encoder_relevant_chunk.append(chunk_relevant_query)
        
    cross_encoder_relevant_chunk = [item for chunk in cross_encoder_relevant_chunk for item in chunk]
    cross_encoder_relevant_chunk = list(dict.fromkeys(cross_encoder_relevant_chunk))
    
    final_chunk = bm25_relevant_chunk + cross_encoder_relevant_chunk
    final_chunk = list(dict.fromkeys(final_chunk))
    return final_chunk

def get_relevant(question):
    docs = search(question)
    
    doc_relevant = ""
    for doc in docs:
        doc_relevant += doc + "\n"
    
    return doc_relevant

def write_prompt(doc_relevant, question):
    
    # Prompt Mistral
    PROMPT = "Dưới đây là ngữ cảnh:\n{doc_relevant}\nHãy trích xuất trong những điều luật đó về nội dung có liên quan đến câu hỏi trắc nghiệm sau:{question}\nỨng với thông tin trích xuất được hãy đưa ra đáp án đúng cho câu hỏi trắc nghiệm, biết rằng các đáp án là chữ số (1,2,3,4,...). Nếu có nhiều đáp án đúng hãy đưa ra đáp án duy nhất. Hãy đưa ra câu trả lời với định dạng sau: Đáp án: <chữ số>\nTrích dẫn từ điều: <số hiệu điều luật>"
    
    input_prompt = PROMPT.format_map(  
    {"doc_relevant": doc_relevant, "question": question}  
)
    return input_prompt

def answer(question):
    # Define the system message
    system_msg = 'Bạn là một trợ lý ảo thông minh, bạn sẽ dùng ngữ cảnh được cung cấp để trả lời câu hỏi trắc nghiệm từ người dùng. Đọc kĩ ngữ cảnh trước khi trả lời câu hỏi và suy nghĩ từng bước một. Tuyệt đối không sử dụng các thông tin khác không nằm trong ngữ cảnh đã cho để trả lời câu hỏi của người dùng.'

    doc_relevant = get_relevant(question)
    # Define the user message
    user_msg = write_prompt(doc_relevant, question)

    response = openai.ChatCompletion.create(model=model,
                                            messages=[{"role": "system", "content": system_msg},
                                             {"role": "user", "content": user_msg}])
    
    return response["choices"][0]["message"]["content"]

def chatbot(question, history=[]):
  output = answer(question)
  history.append((question, output))
  return history, history

demo = gr.Interface(fn=chatbot,
             inputs=["text", "state"],
             outputs=["chatbot", "state"])

demo.queue().launch(share=True)
