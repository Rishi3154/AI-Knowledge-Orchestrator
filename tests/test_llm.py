from app.llm.citations import cited_sources
from app.llm.prompt_builder import PromptBuilder
from app.rag_pipeline import AnswerStream, standalone_question

from tests.conftest import result


class FakeLLM:

    def __init__(self, reply="", tokens=()):
        self.reply = reply
        self.tokens = tokens
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.reply

    def stream(self, prompt):
        self.prompts.append(prompt)
        yield from self.tokens


class FakeRetriever:

    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.results


HISTORY = [
    {"role": "user", "content": "When are the exams?"},
    {"role": "assistant", "content": "Exam 1 is Oct 12, Exam 2 is Nov 30."},
]


# ---------------- citations ----------------

def test_only_cited_sources_are_returned():

    sources = [result("a"), result("b"), result("c")]

    cited = cited_sources("First [3]. Then [1][3].", sources)

    assert [number for number, _ in cited] == [1, 3]


def test_out_of_range_citations_are_ignored():

    assert cited_sources("See [7].", [result("a")]) == []


# ---------------- prompts ----------------

def test_prompt_numbers_each_chunk_with_file_and_page():

    prompt = PromptBuilder.build(
        "What is graded?",
        [
            result("a", "Exams are 50% each.", "syllabus.pdf", 2),
            result("b", "No late work.", "syllabus.pdf", 3),
        ],
    )

    assert "[1] syllabus.pdf, page 2\nExams are 50% each." in prompt
    assert "[2] syllabus.pdf, page 3\nNo late work." in prompt
    assert "What is graded?" in prompt


def test_rewrite_prompt_contains_the_conversation():

    prompt = PromptBuilder.build_rewrite("What about the first?", HISTORY)

    assert "User: When are the exams?" in prompt
    assert "What about the first?" in prompt


# ---------------- follow-up rewriting ----------------

def test_first_question_is_not_rewritten():

    llm = FakeLLM(reply="should not be used")

    assert standalone_question(llm, "When is Exam 1?", []) == "When is Exam 1?"
    assert llm.prompts == []


def test_follow_up_is_rewritten_using_history():

    llm = FakeLLM(reply='"What does Exam 1 cover?"')

    rewritten = standalone_question(llm, "What does the first cover?", HISTORY)

    assert rewritten == "What does Exam 1 cover?"


def test_blank_rewrite_falls_back_to_the_question():

    llm = FakeLLM(reply="   ")

    assert standalone_question(llm, "And the second?", HISTORY) == "And the second?"


def test_runaway_rewrite_falls_back_to_the_question():

    llm = FakeLLM(reply="word " * 200)

    assert standalone_question(llm, "And the second?", HISTORY) == "And the second?"


# ---------------- streaming ----------------

def test_stream_yields_tokens_then_exposes_the_response():

    sources = [result("a")]
    retriever = FakeRetriever(sources)
    llm = FakeLLM(tokens=["Exam ", "1 ", "[1]"])

    stream = AnswerStream(retriever, llm, "When?")

    assert stream.response is None
    assert "".join(stream) == "Exam 1 [1]"
    assert stream.response.answer == "Exam 1 [1]"
    assert stream.response.sources == sources
    assert stream.response.search_query == "When?"


def test_stream_searches_for_the_rewritten_question():

    retriever = FakeRetriever([result("a")])
    llm = FakeLLM(reply="What does Exam 1 cover?", tokens=["ok"])

    stream = AnswerStream(retriever, llm, "What does the first cover?", HISTORY)

    "".join(stream)

    assert retriever.queries == ["What does Exam 1 cover?"]
    assert stream.response.search_query == "What does Exam 1 cover?"


def test_stream_without_documents_asks_for_an_upload():

    stream = AnswerStream(None, FakeLLM(), "When?")

    assert "upload" in "".join(stream).lower()
    assert stream.response.retrieved_chunks == 0


def test_each_stream_keeps_its_own_response():

    first = AnswerStream(FakeRetriever([result("a")]), FakeLLM(tokens=["A"]), "q1")
    second = AnswerStream(FakeRetriever([result("b")]), FakeLLM(tokens=["B"]), "q2")

    "".join(first)
    "".join(second)

    assert first.response.sources[0].chunk_id == "a"
    assert second.response.sources[0].chunk_id == "b"
