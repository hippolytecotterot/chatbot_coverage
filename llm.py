from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

LLM_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert assistant on artificial intelligence and misinformation.
Answer questions based exclusively on the document excerpts provided below.
If the requested information is not present in these excerpts, say so clearly rather than making things up.
If the question is in English, you should answer in English, if the question is in French, you should answer in French.
Always respond concisely, in a few sentences at most. No need to structure your answer.

Relevant document excerpts:
{context}"""


class LLMManager:
    def __init__(self, model: str = LLM_MODEL, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens

    def _format_context(self, docs) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Article")
            source = doc.metadata.get("source", "")
            parts.append(f"[{i} — {title}]({source})\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    def build_chain(self, retriever):
        llm = ChatAnthropic(model=self.model, max_tokens=self.max_tokens)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        return (
            {"context": retriever | self._format_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
