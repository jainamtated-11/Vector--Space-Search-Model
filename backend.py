import zipfile
import os
import re
import math
from collections import defaultdict, Counter

# Global variables
DOCS = []
INDEX = defaultdict(dict)  # Stores term frequencies (tf) for each doc
DOC_FREQS = defaultdict(int)  # Stores document frequencies (df)
TOTAL_DOCS = 0
DOC_LENGTHS = defaultdict(float) # Stores document vector lengths for normalization

def soundex(name):
    """
    Generates the Soundex code for a given name.
    """
    name = name.upper()
    soundex_code = ""
    soundex_map = {
        "BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
        "L": "4", "MN": "5", "R": "6"
    }

    # First letter of the name
    if name:
        soundex_code += name[0]

    # Process remaining letters
    last_digit = ""
    for char in name[1:]:
        digit = ""
        for group, value in soundex_map.items():
            if char in group:
                digit = value
                break
        
        if digit and digit != last_digit:
            soundex_code += digit
        
        last_digit = digit

    # Pad with zeros and truncate
    soundex_code = soundex_code[0] + re.sub(r'0', '', soundex_code[1:])
    soundex_code += "0000"
    return soundex_code[:4]

def preprocess(text):
    """Preprocess the text: lowercase, tokenize, and apply Soundex for names"""
    tokens = re.findall(r"\b\w+\b", text.lower())
    processed_tokens = []
    for token in tokens:
        # Simple heuristic to identify "names" to apply Soundex
        if len(token) > 2 and token.istitle() and token.isalpha():
            processed_tokens.append(soundex(token))
        else:
            processed_tokens.append(token)
    return processed_tokens

def load_corpus(zip_path="corpus-1.zip"):
    """Load all documents from the corpus zip file"""
    global DOCS
    DOCS = []
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Corpus ZIP not found at {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".txt"):
                try:
                    with zip_ref.open(file) as f:
                        DOCS.append(f.read().decode("utf-8"))
                except UnicodeDecodeError:
                    with zip_ref.open(file) as f:
                        DOCS.append(f.read().decode("latin-1"))
                except Exception as e:
                    print(f"Skipping file '{file}' due to an error: {e}")
    return DOCS

def build_index(zip_path="corpus-1.zip"):
    """Build an inverted index with lnc weights and store document vector lengths"""
    global INDEX, DOC_FREQS, TOTAL_DOCS, DOC_LENGTHS
    docs = load_corpus(zip_path)
    TOTAL_DOCS = len(docs)
    INDEX = defaultdict(dict)
    DOC_FREQS = defaultdict(int)
    DOC_LENGTHS = defaultdict(float)

    for i, doc in enumerate(docs):
        tokens = preprocess(doc)
        term_counts = Counter(tokens)
        
        # Calculate TF-ln for documents: 1 + log10(tf)
        doc_weights = {}
        for token, count in term_counts.items():
            tf_ln = 1 + math.log10(count)
            doc_weights[token] = tf_ln
            INDEX[token][i] = tf_ln # Store lnc weight (before normalization)
        
        # Calculate document frequencies
        for token in set(tokens):
            DOC_FREQS[token] += 1
            
        # Calculate document vector length for cosine normalization
        squared_sum = sum(weight**2 for weight in doc_weights.values())
        DOC_LENGTHS[i] = math.sqrt(squared_sum)

def search_query(query):
    """
    Search and rank documents by cosine similarity (lnc.ltc).
    Returns top 10 relevant documents with their IDs and content previews.
    """
    global INDEX, DOC_FREQS, DOCS, DOC_LENGTHS, TOTAL_DOCS
    if not INDEX:
        raise RuntimeError("Index not built yet. Call build_index() first.")

    query_tokens = preprocess(query)
    query_weights = {}
    term_counts = Counter(query_tokens)

    # Calculate TF-ltc for query: 1 + log10(tf) * log10(N/df)
    for token, count in term_counts.items():
        if token in DOC_FREQS:
            tf = 1 + math.log10(count)
            idf = math.log10(TOTAL_DOCS / DOC_FREQS[token])
            query_weights[token] = tf * idf
        else:
            query_weights[token] = 0

    # Calculate query vector length
    query_length = math.sqrt(sum(w**2 for w in query_weights.values()))

    # Calculate cosine similarity for each document
    doc_scores = defaultdict(float)
    relevant_doc_ids = set()
    for token in query_tokens:
        if token in INDEX:
            for doc_id in INDEX[token]:
                relevant_doc_ids.add(doc_id)
    
    for doc_id in relevant_doc_ids:
        dot_product = 0
        for token in query_tokens:
            doc_weight = INDEX.get(token, {}).get(doc_id, 0)
            query_weight = query_weights.get(token, 0)
            dot_product += doc_weight * query_weight
        
        # Cosine similarity = (Dot Product) / (Doc Length * Query Length)
        if DOC_LENGTHS[doc_id] > 0 and query_length > 0:
            cosine_sim = dot_product / (DOC_LENGTHS[doc_id] * query_length)
            doc_scores[doc_id] = cosine_sim
    
    # Sort results by relevance score (descending) and then docID (ascending)
    ranked_results = sorted(
        doc_scores.items(), 
        key=lambda item: (item[1], -item[0]), reverse=True
    )
    
    # Return top 10 results with docID, score, and content
    top_10_results = []
    for doc_id, score in ranked_results[:10]:
        doc_content = DOCS[doc_id]
        top_10_results.append((doc_id, score, doc_content))
    return top_10_results