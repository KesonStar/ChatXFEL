from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

REWRITE_TEMPLATE = """
You are a query rewriting assistant.

Rewrite the user's question into a standalone, unambiguous research query.
Use the conversation history for clarification if needed.

Chat History:
{history}

Original Question:
{question}

Rewritten Search Query:
"""

rewrite_prompt = PromptTemplate(
    template=REWRITE_TEMPLATE,
    input_variables=["history", "question"]
)

def rewrite_query(llm: ChatOllama, question, history):
    formatted_prompt = rewrite_prompt.format(
        history=history if history else "",
        question=question
    )
    
    result = llm.invoke(formatted_prompt)

    # If result is an AIMessage (typical for ChatOllama)
    if hasattr(result, "content"):
        rewritten = result.content
    else:
        # Fallback: convert to text
        rewritten = str(result)

    return rewritten.strip()
