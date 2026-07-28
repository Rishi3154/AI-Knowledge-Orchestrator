from app.retrieval.retriever import Retriever

retriever = Retriever()

results = retriever.search(
    "What is supervised learning?"
)

print(len(results))

print(results[0].metadata)

print(results[0].distance)

print(results[0].content[:300])