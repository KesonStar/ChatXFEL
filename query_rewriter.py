from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

REWRITE_TEMPLATE = """
You are a query rewriting assistant specialized in X-ray Free-Electron Laser (XFEL) research.

Your task: Rewrite the user's question into a standalone, clear, and searchable query optimized for scientific literature retrieval.

Guidelines:
1. Resolve ambiguous references using conversation history (e.g., "it", "that", "this method")
2. Expand common XFEL abbreviations when helpful for search:
   - SHINE → Shanghai High Repetition Rate XFEL and Extreme Light Facility
   - SFX → Serial Femtosecond Crystallography
   - SPI → Single Particle Imaging
   - FEL → Free-Electron Laser
   - SASE → Self-Amplified Spontaneous Emission
   - XFEL → X-ray Free-Electron Laser
   - EuXFEL → European XFEL
   - LCLS → Linac Coherent Light Source
   - SACLA → SPring-8 Angstrom Compact Free Electron Laser
3. Keep technical terms and specific parameter names (wavelength, pulse duration, photon energy, etc.)
4. If the question is already clear and standalone, return it as-is
5. Focus on the scientific intent rather than casual phrasing
6. Include relevant technical context from chat history

Chat History:
{history}

Original Question:
{question}

Rewritten Search Query (return ONLY the rewritten query, no explanations):
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
