

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()   



BASE_URL = "https://apimsdbxcandidate01.azure-api.net/llm"
MODEL = "gpt-5-mini"

PROMPT_TEMPLATE = """You are an expert AI Data Synthesizer. Your task is to receive data snippets retrieved from a semantic search and synthesize them into a cohesive, non-redundant, and well-formatted answer to the user's query.

Please strictly follow these rules:
1. RELIANCE ON DATA: Base your answer *strictly* on the provided retrieved snippets. Do not hallucinate or include outside knowledge.
2. SYNTHESIS & COHESION: Do not just copy-paste the snippets. Weave the relevant facts together into a smooth, cohesive response that directly answers the user's query.
3. NO REDUNDANCY: Eliminate duplicate information if multiple snippets say the same thing.
4. FORMATTING: Use markdown styling (bolding, bullet points, or concise paragraphs) to make the answer easy to read.
5. MISSING INFO: If the retrieved snippets do not contain the answer to the query, politely respond: "Based on the provided documents, I do not have the information to answer this question."

[Input Format]
User Query: {user_query}
Retrieved Snippets: {retrieved_snippets}

[Output Format]
Provide only the final synthesized answer."""

NO_INFO_ANSWER = ("Based on the provided documents, I do not have the "
                  "information to answer this question.")


def generate_report(query: str, snippets: list[dict]) -> str:
    """Build the prompt from the retrieved snippets and synthesize the answer.

    Everything the agent does lives here: format snippets -> build prompt ->
    call the LLM -> return the final text.
    """
    if not snippets:
        return NO_INFO_ANSWER

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    # turn the retriever output into text the LLM can read
    retrieved_snippets = ""
    for i, snippet in enumerate(snippets):
        retrieved_snippets += (
            f"\n[{i}] source: {snippet['programming_name']}"
            f" | overall rank: {snippet['rank']}"
            f" | similarity: {snippet['similarity_score']}\n"
            f"{snippet['content']}\n"
        )

    prompt = PROMPT_TEMPLATE.format(
        user_query=query,
        retrieved_snippets=retrieved_snippets,
    )

    llm = ChatOpenAI(
        model=MODEL,
        api_key=api_key,
        base_url=BASE_URL,
        use_responses_api=True,                   
        default_headers={"api-key": api_key},    
        max_retries=3,
        timeout=60,
    )

    # the Responses API returns a list of blocks (reasoning + text);
    # .text() keeps only the text ones and joins them
    answer = llm.invoke(prompt).text()

    pd.DataFrame([{"knowledge_question": query, "knowledge_answers": answer}]).to_csv(
        "../result_LLM.csv", index=False, encoding="utf-8-sig"
    )

    return answer



