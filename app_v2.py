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

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

db = pd.read_csv('/home/skmlab/data/quangminh/ProjectII/db.csv')
#db = pd.read_csv('/home/skmlab/data/quangminh/ProjectII/db.csv')

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

def remove_duplicate(list1, list2):
    if list1[-1] == list2[0]:
        list2.pop(0)
    return list2

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
    ref_result = ref_lexical + ref_cross_encoder
    ref_result = list(dict.fromkeys(ref_result))
    
    cross_encoder_relevant_chunk = []
    for ref in ref_cross_encoder:
        chunk_relevant_query = db['chunk'][db['ref'] == ref].tolist()
        cross_encoder_relevant_chunk.append(chunk_relevant_query)
        
    cross_encoder_relevant_chunk = [item for chunk in cross_encoder_relevant_chunk for item in chunk]
    cross_encoder_relevant_chunk = list(dict.fromkeys(cross_encoder_relevant_chunk))
    
    final_chunk = bm25_relevant_chunk + cross_encoder_relevant_chunk
    final_chunk = list(dict.fromkeys(final_chunk))
    
    new_final_chunk = [chunk.split("\n\n") for chunk in final_chunk]
    
    no_duplicate_chunk = [new_final_chunk[0]]
    for i in range(1, len(new_final_chunk)):
        new_no_duplicate = remove_duplicate(new_final_chunk[i-1], new_final_chunk[i])
        no_duplicate_chunk.append(new_no_duplicate)

    result = ["\n\n".join(no_duplicate_chunk[i]) for i in range(len(no_duplicate_chunk))]
    
    return result, ref_result[0]
    
# Load model
model_path = "Open-Orca/Mistral-7B-OpenOrca"
#model_path = "SeaLLMs/SeaLLM-7B-Chat"
#model_path = "vilm/vinallama-7b"

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"":0})

def get_relevant(question):
    docs, ref = search(question)
    
    doc_relevant = ""
    for doc in docs:
        doc_relevant += doc + "\n"
    
    return doc_relevant, ref

def write_prompt(doc_relevant, question):
    # Prompt SeaLLM
    BOS_TOKEN = '<s>'
    EOS_TOKEN = '</s>'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    #PROMPT = """You are a multilingual, helpful, respectful and honest assistant. Please always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. Your response should adapt to the norms and customs of the respective language and culture.\n\n Bạn sẽ dùng ngữ cảnh được cung cấp để trả lời câu hỏi từ người dùng. Đọc kĩ ngữ cảnh trước khi trả lời câu hỏi và suy nghĩ từng bước một. Dưới đây là ngữ cảnh:\n{doc_relevant}\nHãy trích xuất trong những điều luật đó về nội dung có liên quan đến câu hỏi sau:"{question}"\n Ứng với câu trả lời thu được hãy trích dẫn số hiệu điều luật mà câu trả lời sử dụng, ví dụ "Câu trả lời: ...\nTrích dẫn từ điều: ..."""

    # Prompt Mistral
    PROMPT = """### Instruction:\n\nBạn là một trợ lý ảo thông minh, bạn sẽ dùng ngữ cảnh được cung cấp để trả lời câu hỏi trắc nghiệm từ người dùng. Đọc kĩ ngữ cảnh trước khi trả lời câu hỏi và suy nghĩ từng bước một. Tuyệt đối không sử dụng các thông tin khác không nằm trong ngữ cảnh đã cho để trả lời câu hỏi của người dùng. Dưới đây là ngữ cảnh:\n{doc_relevant}\nHãy trích xuất trong những điều luật đó về nội dung có liên quan đến câu hỏi trắc nghiệm sau:"{question}"\nỨng với thông tin trích xuất được hãy đưa ra đáp án đúng cho câu hỏi trắc nghiệm, biết rằng các đáp án là chữ số (1,2,3,4,...). Nếu có nhiều đáp án đúng hãy đưa ra đáp án duy nhất. Hãy đưa ra câu trả lời với định dạng sau: "Đáp án: <chữ số>\nTrích dẫn từ điều: <điều luật>"\n\n### Response:""" 
    
    input_prompt = PROMPT.format_map(  
    {"doc_relevant": doc_relevant, "question": question}  
)
    return input_prompt

def generate(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt")
    
    outputs = model.generate(  
        inputs=input_ids["input_ids"].to("cuda"),  
        attention_mask=input_ids["attention_mask"].to("cuda"),  
        do_sample=True,  
        max_new_tokens=1024, # 1024  
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,  
    )  

    return outputs

def answer(question):
    doc_relevant, ref = get_relevant(question)
    input_prompt = write_prompt(doc_relevant, question)
    outputs = generate(input_prompt)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
    response = response.split("### Response:")[1]
    
    ref = "\n" + "Trích dẫn từ: " + ref
    
    response = response + ref
    
    return response.strip()

def chatbot(question, history=[]):
  output = answer(question)
  history.append((question, output))
  return history, history

demo = gr.Interface(fn=chatbot,
             inputs=["text", "state"],
             outputs=["chatbot", "state"])

demo.queue().launch(share=True)
