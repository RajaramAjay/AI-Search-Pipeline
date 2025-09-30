# src/prompt_engg.py
from typing import List

class PromptEngineering:
    """
    A plug-and-play class for generating prompts for different LLM techniques:
    zero-shot, few-shot, chain-of-thought (CoT), self-consistency.
    Supports RAG and generic prompting.
    This class now **returns only the prompt text**, without calling the LLM.
    """

    def __init__(self):
        pass  # No LLM callable needed anymore

    # ----------------- Zero-Shot -----------------
    def zero_shot(self, instruction: str, query: str = None) -> str:
        """
        Generate a zero-shot prompt.
        If query is provided, append it to instruction.
        """
        prompt = instruction
        if query:
            prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    # ----------------- Few-Shot -----------------
    def few_shot(self, instruction: str, examples: List[str] = None, query: str = None) -> str:
        """
        Generate a few-shot prompt with optional examples and query.
        """
        prompt = instruction
        if examples:
            prompt += "\n\n" + "\n".join(examples)
        if query:
            prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    # ----------------- Chain-of-Thought -----------------
    def chain_of_thought(self, instruction: str, query: str = None) -> str:
        """
        Generate a chain-of-thought prompt (step-by-step reasoning).
        """
        prompt = instruction + "\nThink step by step."
        if query:
            prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    # ----------------- Self-Consistency -----------------
    def self_consistency(self, instruction: str, query: str = None, num_samples: int = 5) -> List[str]:
        """
        Generate multiple chain-of-thought prompts for self-consistency.
        Returns a list of identical prompts that can be used for multiple LLM calls.
        """
        prompts = []
        base_prompt = instruction + "\nThink step by step."
        if query:
            base_prompt += f"\nQuestion: {query}\nAnswer:"
        for _ in range(num_samples):
            prompts.append(base_prompt)
        return prompts

# ----------------- Example Usage -----------------
if __name__ == "__main__":
    pe = PromptEngineering()

    base_instruction = "Answer strictly based on the provided context.\n\nContext: The context goes here."

    # Zero-shot
    print("Zero-Shot Prompt:\n", pe.zero_shot(base_instruction, "Explain Pydantic."))

    # Few-shot
    examples = [
        "Q: Hello\nA: Bonjour",
        "Q: Good morning\nA: Bonjour"
    ]
    print("Few-Shot Prompt:\n", pe.few_shot(base_instruction, examples, "I love AI"))

    # Chain-of-thought
    print("Chain-of-Thought Prompt:\n", pe.chain_of_thought(base_instruction, "If 3+2=5, what is 7+3?"))

    # Self-consistency
    print("Self-Consistency Prompts:\n", pe.self_consistency(base_instruction, "If 3+2=5, what is 7+3?", num_samples=3))
