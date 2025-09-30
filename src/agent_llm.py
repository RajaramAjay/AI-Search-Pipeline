#src/agent_llm.py
from langchain_community.llms import Ollama  # updated import
import toml, os, json, re
from logger_setup import setup_logger

logger = setup_logger(__name__)

# Load config
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)
model_name = config['LLMmodel']['model']

# Initialize Ollama
ollama_llm = Ollama(model=model_name)
logger.info(f"Ollama initialized with model: {model_name}")


def call_real_llm(prompt: str) -> dict:
    """
    Calls Ollama LLM with the given prompt and returns a dict with 'score' and 'reasoning'.
    Handles:
    - Proper JSON returned by LLM
    - Plain text output
    - Messy JSON-like text with numeric score anywhere
    """
    try:
        # Call the LLM
        raw_response = ollama_llm.invoke(prompt)
        # print(f"Raw response from LLM: {raw_response}")
        text = raw_response.strip()

        # Try parsing JSON directly
        try:
            parsed = json.loads(text)
            # Ensure keys exist
            score = parsed.get("score", 0.0)
            reasoning = parsed.get("reasoning", text)
            return {"score": float(score), "reasoning": reasoning}

        except json.JSONDecodeError:
            # Look for numeric score anywhere in the text using regex
            score_match = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', text)
            score = float(score_match.group(1)) if score_match else 0.0

            # Remove score part from reasoning if present
            reasoning = re.sub(r'"?score"?\s*[:=]\s*[0-9]*\.?[0-9]+', '', text, flags=re.IGNORECASE).strip()

            # Wrap in dict
            return {"score": score, "reasoning": reasoning}

    except Exception as e:
        logger.warning(f"LLM returned non-JSON response; wrapping into JSON with default score=0.0")
        logger.error(f"LLM call failed: {e}", exc_info=True)
        return {"score": 0.0, "reasoning": f"LLM call failed: {e}"}
    



    
# ----------------- Sample Usage -----------------
if __name__ == "__main__":
    logger.info("Starting sample test for agent_llm LLM evaluation")
    
    sample_prompt = """
    Evaluate the following answer for correctness and clarity.
    Question: What is Pydantic?
    Answer: Ajay is a very good boy who is learning Python and LangChain. He is very curious and loves to learn new things.
    Provide JSON output with fields: score (0-1) and reasoning (text explanation).
    """
    response = call_real_llm(sample_prompt)
    print("\n=== LLM Evaluation Response ===")
    print(json.dumps(response, indent=2))
