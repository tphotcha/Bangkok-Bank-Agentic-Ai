import os
import re

import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ---------1. Load ------------
loader = TextLoader("../knowlege_base.txt", encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} Documents, {len(documents[0].page_content)} characters")


# --------2. Chunk ------------
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "title"), ("##", "heading")],
    strip_headers=True,
)

chunks = []
for doc in documents:
    for chunk in splitter.split_text(doc.page_content):
        # splitter เก็บหัวข้อไว้ใน chunk.metadata -> อ่านออกมาก่อนเขียนทับ
        heading = chunk.metadata.get("heading", "").strip()
        match = re.match(r"(.+?)\s*[—–-]\s*Rank\s*(\d+)\s*$", heading)

        chunk.metadata = {
            "programming_name": match.group(1).strip() if match else "",
            "rank_score": int(match.group(2)) if match else 0,
        }
        chunks.append(chunk)

print(f"Have {len(chunks)} chunks")


# for i, chunk in enumerate(chunks):
#     print(f"\n--- chunk {i} | {len(chunk.page_content)} chars ---")
#     print("metadata:", chunk.metadata)
#     print(chunk.page_content[:200].replace("\n", " "), "...")


# --- 3. Embeddings -------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)


# --- 4. Vector store ------------
INDEX_DIR = "../knowledge_index"

# load_local does NOT restore these two settings -> must pass them again,
# otherwise the loaded store silently falls back to euclidean distance
faiss_kwargs = {
    "normalize_L2": True,
    "distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT,
}

if os.path.isdir(INDEX_DIR):
    print(f"Index found at {INDEX_DIR} -> loading (skip embedding)")
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
        **faiss_kwargs,
    )
else:
    vectorstore = FAISS.from_documents(chunks, embeddings, **faiss_kwargs)
    vectorstore.save_local(INDEX_DIR)


# --- 5. semantic search ------------


def vector_search(query: str, k: int = 10) -> list[dict]:

    results = vectorstore.similarity_search_with_score(query, k=k)

    snippets = [
        {
            "programming_name": doc.metadata["programming_name"],
            "rank": doc.metadata["rank_score"],
            "similarity_score": round(float(score), 4),
            "content": doc.page_content,
        }
        for doc, score in results
    ]

    # save the ranking of this search (drop the long content column)
    df = pd.DataFrame(snippets).drop(columns=["content"])
    df.to_csv("../results.csv", index=False, encoding="utf-8-sig")

    return snippets

 
