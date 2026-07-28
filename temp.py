from app.embeddings.embedder import OllamaEmbedder

embedder = OllamaEmbedder()

embedding = embedder.embed("Machine Learning")

print(type(embedding))

print(len(embedding))

print(embedding[:5])