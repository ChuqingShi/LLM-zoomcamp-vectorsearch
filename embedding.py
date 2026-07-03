from tqdm.auto import tqdm
import numpy as np


def preprocess_documents(documents, course=None):
    texts = []
    docs = []
    for doc in documents:
        if course is None or doc["course"] == course:
            text = doc["question"] + " " + doc["answer"]
            texts.append(text)
            docs.append(doc)
    return docs,texts


def get_embedding(texts, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
        X = np.array(embeddings)
    return X


def vector_search(X, documents, query, model, top_k=5):
    query_embedding = model.encode(query)
    scores = X.dot(query_embedding)
    top_indices = np.argsort(-scores)[:top_k]
    results = [documents[i] for i in top_indices]
    return results

