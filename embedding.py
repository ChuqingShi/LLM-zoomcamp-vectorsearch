from tqdm.auto import tqdm
import numpy as np


def preprocess_documents(documents, course=None):
    print("Preprocessing documents...")
    texts = []
    if course is None:
        for doc in documents:
            text = doc["question"] + " " + doc["answer"]
            texts.append(text)
        return documents, texts
    else:
        print("Filtering documents for course:", course)
        docs = []
        for doc in documents:
            if doc["course"] == course:
                text = doc["question"] + " " + doc["answer"]
                texts.append(text)
                docs.append(doc)
        print(f"Found {len(docs)} documents for course:", course)
        return docs, texts


def get_embedding(texts, model, batch_size=32):
    embeddings = []
    print("Generating embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    X = np.array(embeddings)
    return X


def vector_search(X, documents, query, model, top_k=5):
    print("Performing vector search...")
    query_embedding = model.encode(query)
    scores = X.dot(query_embedding)
    top_indices = np.argsort(-scores)[:top_k]
    top_scores = scores[top_indices]
    results = [documents[i] for i in top_indices]
    print(f"Found {len(results)} results for query: {query}")
    return results, top_scores

