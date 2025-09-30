# src/llm_response.py
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from src.ensemble_retrieval import EnsembleRetrieverSystem
from src.embeddings import SentenceTransformerEmbeddings
from src.prompt_engg import PromptEngineering
import toml, os
from logger_setup import setup_logger
import time
from src.langraph import evaluate_llm_output
from src.agent_llm import call_real_llm

# Logger setup
logger = setup_logger(__name__)

# Load configuration from TOML file
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
logger.debug(f"Loading configuration from {config_path}")
config = toml.load(config_path)


class LLMInvoke:
    def __init__(self):
        """
        Initialize LLMInvoke with configuration.
        """
        logger.info("Initializing LLMInvoke")
        self.faiss_path = config['faiss']['path']
        logger.debug(f"FAISS index path: {self.faiss_path}")

        # Embeddings
        self.embeddings = SentenceTransformerEmbeddings()
        logger.debug("Initialized SentenceTransformerEmbeddings")

        # Initialize Ollama LLM
        self.model = config['LLMmodel']['model']
        logger.info(f"Initializing Ollama with model {self.model}")
        self.llm = Ollama(model=self.model)

        # Prompt Engineering (no LLM required)
        self.prompt_engineer = PromptEngineering()

    def llm_response(self, query, prompt_mode="zero_shot", self_consistency=False,
                     reference_answer=None, run_evaluation=False):
        """
        Retrieve relevant documents and generate LLM response based on prompt engineering.
        Args:
            query (str): User query
            prompt_mode (str): 'zero_shot', 'few_shot', 'chain_of_thought'
            self_consistency (bool): If True, generate multiple prompts for self-consistency
            num_samples (int): Number of prompts for self-consistency
            reference_answer (str): Optional reference for evaluation
            run_evaluation (bool): If True, run evaluation
        Returns:
            dict: {
                'answer': str,
                'sources': list of dicts,
                'question': query
            }
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}'")

        try:
            # Load FAISS index
            logger.debug(f"Loading FAISS index from {self.faiss_path}")
            load_start = time.time()
            doc_index = FAISS.load_local(
                self.faiss_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.debug(f"FAISS index loaded in {time.time() - load_start:.3f}s")

            # Retrieve documents
            logger.info("Retrieving documents using ensemble retrieval")
            retrieval_start = time.time()
            final_docs = EnsembleRetrieverSystem.get_ensemble_results(query, doc_index, doc_index.index)
            logger.info(f"Retrieved {len(final_docs)} documents in {time.time() - retrieval_start:.3f}s")

            documents = [
                {
                    'page_content': doc.page_content,
                    'metadata': doc.metadata,
                    'average_relevance_score': float(score),
                    'retriever_vote_count': count
                }
                for doc, score, count in final_docs
            ]

            for i, doc in enumerate(documents):
                logger.info(f"Document {i+1}: {doc['metadata'].get('id', 'N/A')} - "
                            f"Score: {doc['average_relevance_score']:.4f}, "
                            f"Votes: {doc['retriever_vote_count']}")

            # Build context
            context = "\n".join([doc["page_content"] for doc in documents])
            base_instruction = f"Answer brefly based on the provided context.\n\nContext: {context}"

            # ------------------ Generate Prompt ------------------
            if prompt_mode == "chain_of_thought":
                if self_consistency:
                    prompts_list = self.prompt_engineer.self_consistency(base_instruction, query, num_samples=3)
                else:
                    prompts_list = [self.prompt_engineer.chain_of_thought(base_instruction, query)]
            else:  # zero_shot
                prompts_list = [self.prompt_engineer.zero_shot(base_instruction, query)]

            logger.info(f"Generated {prompts_list} prompts for LLM invocation")    

            # ------------------ Call LLM ------------------
            answers = []
            for prompt in prompts_list:
                llm_start = time.time()
                answer = self.llm(prompt)
                logger.info(f"LLM call completed in {time.time() - llm_start:.3f}s")
                answers.append(answer)

            # If self-consistency, pick the most frequent answer
            if self_consistency and len(answers) > 1:
                answer_counts = {}
                for ans in answers:
                    answer_counts[ans] = answer_counts.get(ans, 0) + 1
                final_answer = max(answer_counts, key=answer_counts.get)
            else:
                final_answer = answers[0]

            # Build sources list
            sources = []
            for doc in documents:
                metadata = doc.get("metadata", {})
                relevance = doc.get("average_relevance_score", 0.0)

                source_info = {"relevance": relevance}
                for key in ["source", "url", "page_number"]:
                    if key in metadata:
                        source_info[key] = metadata[key]
                if any(k in source_info for k in ["source", "url", "page_number"]):
                    sources.append(source_info)

            # Prepare final output
            llm_output = {
                'answer': final_answer,
                'sources': sources,
                'question': query
            }

            # Run evaluation if needed
            if run_evaluation and reference_answer:
                eval_results = evaluate_llm_output(
                    llm_output=llm_output,
                    reference_answer=reference_answer,
                    llm_callable=call_real_llm
                )
                logger.info(f"Evaluation Results: {eval_results}")
                return eval_results

            logger.info(f"Query processing completed in {time.time() - start_time:.3f}s")
            return llm_output

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                'answer': f"Error processing query: {str(e)}",
                'sources': [],
                'question': query
            }


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    llm_invoker = LLMInvoke()

    query = "What is Pydantic, and how do I use it?"
    reference = "Pydantic is a Python library for data validation and settings management using Python type hints."

    result = llm_invoker.llm_response(
        query,
        prompt_mode="chain_of_thought",
        self_consistency=True,
        num_samples=3,
        reference_answer=reference,
        run_evaluation=True
    )

    print("\n=== Full Evaluation Result ===")
    print(result)
