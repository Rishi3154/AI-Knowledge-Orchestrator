from app.rag_pipeline import RAGPipeline

print("=" * 60)
print("AI Knowledge Orchestrator")
print("Type 'exit' to quit")
print("=" * 60)

rag = RAGPipeline()

while True:

    question = input("\nYou: ")

    if question.lower() in {"exit", "quit"}:
        print("\n" + "=" * 60)
        print("Thanks for using AI Knowledge Orchestrator!")
        print("Goodbye! 👋")
        print("=" * 60)
        break

    print("\nAssistant:\n")

    for token in rag.stream(question):
        print(token, end="", flush=True)

    response = rag.last_response
    print()

    print("\n" + "=" * 60)
    print("📚 Sources")
    print("=" * 60)

    seen = set()

    for source in response.sources:

        key = (
            source.metadata["source"],
            source.metadata["page"],
        )

        if key in seen:
         continue

        seen.add(key)

        print(
        f"- {source.metadata['source']} "
        f"(Page {source.metadata['page']})"
        )

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    print(f"Retrieved Chunks : {response.retrieved_chunks}")
    print(f"Latency          : {response.latency:.2f}s")