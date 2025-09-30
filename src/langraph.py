# src/langraph.py
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, ValidationError
import json
from typing import Optional
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import evaluate  # METEOR metric
from src.agent_llm import call_real_llm
from src.prompt_engg import PromptEngineering
from typing import List

# ------------------- Pydantic Schemas -------------------
class EvaluatedAnswer(BaseModel):
    """
    Represents a single LLM answer evaluation structure.
    """
    answer: str
    sources: list = Field(default_factory=list)
    question: Optional[str] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None


class StateSchema(BaseModel):
    """
    Represents the state of a single evaluation workflow step.
    """
    llm_output: dict
    reference_answer: Optional[str] = None
    validated_output: Optional[dict] = None
    validation_passed: bool = False
    validation_error: Optional[str] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None
    meteor: Optional[float] = None
    llm: Optional[callable] = None  # Optional LLM callable for evaluation

    # Allow arbitrary types (like function references) in Pydantic
    model_config = {"arbitrary_types_allowed": True}


# ------------------- Node Functions -------------------
# Node 1 Validate LLM Output
def validate_node(state: StateSchema):
    """
    Node 1: Validate the structure of LLM output using Pydantic.
    """
    print("\n[Node] validate_node: Validating LLM output structure...")
    try:
        validated = EvaluatedAnswer(**state.llm_output)
        state.validated_output = validated.model_dump()  # Pydantic V2 method
        state.validation_passed = True
        print("Validation Passed.")
        print(f"Validated Output: {state.validated_output}")
    except ValidationError as e:
        state.validation_error = str(e)
        state.validation_passed = False
        print("Validation Failed.")
        print(f"Error: {state.validation_error}")
    return state

# Node 2 Quantitative Evaluation
def quantitative_evaluation(state: StateSchema):
    """
    Node 2: Compute BLEU, ROUGE-L, and METEOR if reference answer is provided.
    """
    print("\n[Node] quantitative_evaluation: Starting quantitative evaluation...")
    if state.reference_answer:
        candidate = state.llm_output.get("answer", "").split()
        reference = state.reference_answer.split()

        # BLEU score
        state.bleu = sentence_bleu([reference], candidate)
        print(f"BLEU score: {state.bleu}")

        # ROUGE-L score
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(state.reference_answer, state.llm_output.get("answer", ""))
        state.rouge_l = rouge_scores['rougeL'].fmeasure
        print(f"ROUGE-L score: {state.rouge_l}")

        # METEOR score
        meteor_metric = evaluate.load('meteor')
        state.meteor = meteor_metric.compute(
            predictions=[state.llm_output.get("answer", "")],
            references=[state.reference_answer]
        )['meteor']
        print(f"METEOR score: {state.meteor}")
    else:
        print("No reference answer provided. Skipping quantitative evaluation.")
    return state

# Node 3 LLM-based Evaluation
def llm_based_evaluation(state: StateSchema):
    """
    Node 3: Evaluate the answer for correctness, clarity, and reasoning.
    Uses predefined few-shot examples with structured JSON responses.
    """
    print("\n[Node] llm_based_evaluation: Starting LLM-based evaluation...")

    if state.llm is None:
        state.reasoning = "LLM evaluation skipped: no LLM provided."
        print(state.reasoning)
        return state

    answer_text = state.llm_output.get("answer", "")
    question = state.llm_output.get("question", "N/A")

    # Base instruction for evaluation
    instruction = (
        "You are an AI evaluator. Evaluate the following LLM answer strictly.\n"
        "Instructions:\n"
        "1. Evaluate correctness, relevance, and clarity.\n"
        "2. Return ONLY JSON with fields: 'score' (0-1), 'reasoning' (text explanation).\n"
        "3. Do NOT add commentary, markdown, or examples outside the JSON.\n"
    )

    # Predefined few-shot examples in JSON format
    examples = [
        """Example of correct response:
    {
    "score": 0.85,
    "reasoning": "The answer is mostly correct and relevant, but could be clearer in explaining the details of Pydantic."
    }""",
        """Example of correct response:
    {
    "score": 1.0,
    "reasoning": "The answer is completely correct, relevant, and very clear."
    }"""
    ]

    # Initialize prompt engineering
    pe = PromptEngineering()

    # Construct prompt using few-shot
    prompt = pe.few_shot(
        instruction=instruction,
        examples=examples,
        query=f"Question: {question}\nAnswer: {answer_text}"
    )

    print(f"Prompt sent to evaluation LLM:\n{prompt}\n")

    # Call LLM
    try:
        result = state.llm(prompt)
        if 'score' in result:
            state.score = float(result['score'])
        if 'reasoning' in result:
            state.reasoning = str(result['reasoning'])
        print(f"LLM-based evaluation result: {result}")
    except Exception as e:
        state.reasoning = f"LLM evaluation failed: {e}"
        state.score = 0.0
        print(state.reasoning)

    return state

# ------------------- Build Langraph Workflow -------------------
graph = StateGraph(state_schema=StateSchema)

# Add nodes
graph.add_node("validate", validate_node)
graph.add_node("quant_eval", quantitative_evaluation)
graph.add_node("llm_eval", llm_based_evaluation)

# Define workflow edges
graph.set_entry_point("validate")
graph.add_edge("validate", "quant_eval")
graph.add_edge("quant_eval", "llm_eval")
graph.add_edge("llm_eval", END)

# Compile workflow
workflow = graph.compile()


# ------------------- Plug-and-Play Function -------------------
def evaluate_llm_output(llm_output: dict, reference_answer=None, llm_callable=None):
    """
    Evaluate an LLM output using the workflow.
    Returns a dictionary with quantitative and qualitative metrics.
    """
    initial_state = StateSchema(
        llm_output=llm_output,
        reference_answer=reference_answer,
        llm=llm_callable
    )
    result = workflow.invoke(initial_state)
    return result  # Already a dict


# ------------------- Sample Test -------------------
if __name__ == "__main__":
    # Example LLM output
    # sample_response = {'answer': "Pydantic is a framework for building robust and predictable data models in Python. It provides classes that can be used to define and validate complex data structures. Pydantic uses TypeCheckingExtensions for defining TypedDictionaries which are useful for specifying the structure of objects.\n\nTo answer your question, here's how you can use Pydantic:\n\n1. Import the necessary modules from `typing_extensions` instead of using `typing`.\n\n2. Define a TypedDict class to represent the structure of your data, such as a personal information schema.\n3. Optional: define fields that should be optional by specifying them with the `Optional` type.\n\nHere is an example:\n\n```python\nfrom typing_extensions import TypedDict\n\nclass PersonalInfo(TypedDict):\n    name: str\n    age: int\n    occupation: Optional[str]\n```\n\n4. You can then use this class to define a Pydantic model, for example as a function that extracts personal information from text.\n\nHere is an example:\n\n```python\nfrom typing import List\n\ndef extract_personal_info(text: str) -> PersonalInfo:\n    # This function should return a dictionary or any other data structure that represents the extracted personal info.\n    pass\n```\n\n5. To validate the input, Pydantic will raise an error if the required fields are missing.\n\n6. You can then use this model to generate responses based on the inputs and expectations defined in your schema.", 'sources': [{'relevance': 3.3286054134368896, 'source': 'https://python.langchain.com/v0.2/docs/how_to/structured_output/?source=post_page-----a63c2a0c440a---------------------------------------#__docusaurus_skipToContent_fallback'}, {'relevance': 2.6987629244404454, 'source': 'https://python.langchain.com/v0.2/docs/tutorials/extraction/'}, {'relevance': 2.563036371449955, 'source': 'https://python.langchain.com/v0.2/docs/how_to/structured_output/?source=post_page-----a63c2a0c440a---------------------------------------#__docusaurus_skipToContent_fallback'}]}
    sample_response = {
    "answer": "Ajay is good boy",
    "sources": [
        {"source": "https://www.python.org/", "relevance": 3.0},
        {"source": "https://realpython.com/", "relevance": 2.5}
    ],
    "question": "What is Pydantic?"
}

    # Reference answer (optional)
    reference = "Pydantic is a Python library used to define and validate structured data models."

    # Run evaluation workflow
    print("\n=== Starting Evaluation Workflow ===")
    evaluation = evaluate_llm_output(sample_response,reference_answer=reference, llm_callable=call_real_llm)
    # evaluation = evaluate_llm_output(sample_response, llm_callable=call_real_llm)
    print("\n=== Evaluation Workflow Completed ===")

    # Safely print final evaluation result excluding the LLM callable
    print("\n=== Final Evaluation Result ===")
    print(json.dumps({k: v for k, v in evaluation.items() if k != 'llm'}, indent=2))
