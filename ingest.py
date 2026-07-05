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
