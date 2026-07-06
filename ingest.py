import requests
from minsearch import VectorSearch
from sqlitesearch import VectorSearchIndex


def load_faq_data():
    docs_url = 'https://datatalks.club/faq/json/courses.json'
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = 'https://datatalks.club/faq'

    for course in courses_raw:
        course_url = f'{url_prefix}{course["path"]}'
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)

    return documents


def build_vector_index(X, documents):
    index = VectorSearch(
        keyword_fields=['course']
    )
    index.fit(X, documents)
    return index


def build_sqlite_vector_index(X, documents, db_path="faq_vectors.db"):
    index = VectorSearchIndex(
        keyword_fields=['course'],
        mode="ivf",
        db_path=db_path,
    )
    index.fit(X, documents)
    index.close()
    return index


def _iter_batches(items, batch_size=32):
    for i in range(0, len(items), batch_size):
        print(f"Yielding batch {i // batch_size + 1} of {(len(items) - 1) // batch_size + 1}")
        yield items[i:i + batch_size]
        

def iter_faq_batches(batch_size=32):
    docs_url = "https://datatalks.club/faq/json/courses.json"
    courses = requests.get(docs_url).json()
    
    url_prefix = "https://datatalks.club/faq"

    for course in courses:
        print("Processing course:", course['course'])
        course_data = requests.get(f"{url_prefix}{course['path']}").json()

        for batch in _iter_batches(course_data, batch_size):
            yield batch
