# src/ensemble_retrieval.py
import numpy as np
from collections import defaultdict
from sentence_transformers.cross_encoder import CrossEncoder
from src.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from logger_setup import setup_logger
import time
import os
import toml
# Class variables for configuration
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.toml'))
config = toml.load(config_path)
model_path = config['embeddings']['encoder_model']

# Set up logger as a class variable
logger = setup_logger(__name__)

class EnsembleRetrieverSystem:
    """
    Object-oriented implementation of an ensemble retrieval system that combines
    multiple retrieval methods to provide higher quality search results.
    """
    def __init__(self):
        """
        Initialize the EnsembleRetrieverSystem.
        Note: Most functionality is handled by static methods, so this constructor is minimal.
        """
        pass

    @staticmethod
    def reciprocal_rank_fusion(results_lists, k=60):
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results_lists: List of ranked document lists, each containing (doc, score) tuples
            k: Constant to prevent items with very low scores from having too much impact
        
        Returns:
            Combined ranked list with (doc, score) tuples
        """
        logger.info(f"Starting reciprocal rank fusion with {len(results_lists)} result lists")
        start_time = time.time()
        
        rrf_scores = {}
        doc_mapping = {}
        
        for i, results in enumerate(results_lists):
            logger.debug(f"Processing result list {i+1}/{len(results_lists)} with {len(results)} documents")
            for rank, (doc, _) in enumerate(results):
                doc_id = f"{doc.metadata['id']}_{doc.metadata['content_hash']}"
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
                doc_mapping[doc_id] = doc
                
        sorted_doc_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        result = [(doc_mapping[doc_id], rrf_scores[doc_id]) for doc_id in sorted_doc_ids]
        
        logger.info(f"Finished reciprocal rank fusion in {time.time() - start_time:.3f}s. "
                    f"Combined {len(doc_mapping)} unique documents.")
        return result

    @staticmethod
    def weighted_voting_ensemble(results_lists, method_weights=None):
        """
        Combine multiple result lists using weighted voting ensemble method.
        
        Args:
            results_lists: List of ranked document lists, each containing (doc, score) tuples
            method_weights: Weight for each retrieval method (defaults to equal weights)
        
        Returns:
            Combined ranked list with (doc, score) tuples
        """
        logger.info(f"Starting weighted voting ensemble with {len(results_lists)} result lists")
        start_time = time.time()
        
        if method_weights is None:
            method_weights = [1.0] * len(results_lists)
            logger.debug("Using default equal weights for all methods")
        else:
            logger.debug(f"Using custom weights: {method_weights}")
            
        ensemble_scores = defaultdict(float)
        normalization_factors = defaultdict(float)
        doc_mapping = {}
        
        # Process each result list with its corresponding weight
        for i, (results, weight) in enumerate(zip(results_lists, method_weights)):
            logger.debug(f"Processing method {i+1} with weight {weight}: {len(results)} results")
            # Normalize scores within each result list
            max_score = max((score for _, score in results), default=1.0)
                
            for doc, score in results:
                doc_id = f"{doc.metadata['id']}_{doc.metadata['content_hash']}"
                normalized_score = score / max_score if max_score > 0 else 0
                ensemble_scores[doc_id] += normalized_score * weight
                normalization_factors[doc_id] += weight
                doc_mapping[doc_id] = doc
                
        # Normalize final scores by total weight
        for doc_id in ensemble_scores:
            if normalization_factors[doc_id] > 0:
                ensemble_scores[doc_id] /= normalization_factors[doc_id]
                
        sorted_doc_ids = sorted(ensemble_scores, key=ensemble_scores.get, reverse=True)
        result = [(doc_mapping[doc_id], ensemble_scores[doc_id]) for doc_id in sorted_doc_ids]
        
        logger.info(f"Finished weighted voting ensemble in {time.time() - start_time:.3f}s. "
                    f"Combined {len(doc_mapping)} unique documents.")
        return result

    @staticmethod
    def reranker_ensemble(query, results_lists, cross_encoder):
        """
        Reranks documents from multiple result lists using a cross-encoder model.
        
        Args:
            query: Search query
            results_lists: List of ranked document lists, each containing (doc, score) tuples
            cross_encoder: Cross-encoder model for relevance scoring
            
        Returns:
            Reranked list of (doc, score) tuples
        """
        logger.info("Starting cross-encoder reranking of documents")
        start_time = time.time()
        
        # Collect unique documents from all result lists
        unique_docs = {}
        for i, results in enumerate(results_lists):
            logger.debug(f"Processing result list {i+1}/{len(results_lists)} with {len(results)} documents")
            for doc, _ in results:
                doc_id = f"{doc.metadata['id']}_{doc.metadata['content_hash']}"
                unique_docs[doc_id] = doc
                    
        # Prepare query-document pairs for cross-encoder scoring
        logger.debug(f"Creating {len(unique_docs)} query-document pairs for scoring")
        pairs = [[query, doc.page_content] for doc in unique_docs.values()]
        
        # Score pairs using cross-encoder
        logger.debug("Running cross-encoder prediction")
        scores = cross_encoder.predict(pairs)
        
        # Create and sort scored results
        scored_results = list(zip(unique_docs.values(), scores))
        result = sorted(scored_results, key=lambda x: x[1], reverse=True)
        
        logger.info(f"Finished cross-encoder reranking in {time.time() - start_time:.3f}s. "
                    f"Reranked {len(unique_docs)} documents.")
        return result

    @staticmethod
    def get_ensemble_results(query, doc_index, faiss_index):
        """
        Performs multi-strategy document retrieval and ensemble ranking.
        
        Args:
            query: Search query
            doc_index: Document index
            faiss_index: FAISS vector index
            
        Returns:
            List of top 5 documents with scores
        """
        logger.info(f"Starting ensemble retrieval for query: '{query}'")
        overall_start_time = time.time()
        
        # Initialize components
        logger.debug("Initializing embedding model and cross-encoder")
        embeddings = SentenceTransformerEmbeddings()
        cross_encoder = CrossEncoder(model_path)
        
        # Generate query embedding once for multiple uses
        logger.debug("Generating query embedding")
        query_embedding = embeddings.embed_query(query)
        
        # Collect results from multiple methods
        results_lists = []
        
        # Method 1: FAISS Search
        logger.debug("Running FAISS vector search")
        method_start = time.time()
        D, I = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=5)
        method1_results = [
            (doc_index.docstore._dict[list(doc_index.docstore._dict.keys())[idx]], np.exp(-distance))
            for idx, distance in zip(I[0], D[0]) if idx >= 0
        ]
        results_lists.append(method1_results)
        logger.debug(f"FAISS search completed in {time.time() - method_start:.3f}s with {len(method1_results)} results")
        
        # Method 2: MMR (Maximum Marginal Relevance)
        logger.debug("Running MMR search")
        method_start = time.time()
        mmr_results = [(doc, 1.0) for doc in doc_index.max_marginal_relevance_search(query, k=5)]
        results_lists.append(mmr_results)
        logger.debug(f"MMR search completed in {time.time() - method_start:.3f}s with {len(mmr_results)} results")
        
        # Method 3: Similarity Search
        logger.debug("Running similarity search")
        method_start = time.time()
        similarity_results = [
            (doc, np.exp(-score)) 
            for doc, score in doc_index.similarity_search_with_score(query, k=5)
        ]
        results_lists.append(similarity_results)
        logger.debug(f"Similarity search completed in {time.time() - method_start:.3f}s with {len(similarity_results)} results")
        
        # Method 4: Cross-encoder Reranking
        logger.debug("Running cross-encoder reranking on extended results")
        method_start = time.time()
        # Get more documents for reranking
        D, I = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=10)
        pairs = []
        docs = []
        for idx in I[0]:
            if idx >= 0:
                doc = doc_index.docstore._dict[list(doc_index.docstore._dict.keys())[idx]]
                pairs.append([query, doc.page_content])
                docs.append(doc)
        scores = cross_encoder.predict(pairs)
        ce_results = list(zip(docs, scores))
        results_lists.append(ce_results)
        logger.debug(f"Cross-encoder reranking completed in {time.time() - method_start:.3f}s with {len(ce_results)} results")
        
        # Method 5: Ensemble Retriever (BM25 + FAISS)
        logger.debug("Running BM25 + FAISS ensemble retriever")
        method_start = time.time()
        documents = list(doc_index.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(documents, k=5)
        faiss_retriever = doc_index.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6])
        ensemble_results = [(doc, 1.0) for doc in ensemble_retriever.invoke(query)]
        results_lists.append(ensemble_results)
        logger.debug(f"BM25+FAISS ensemble completed in {time.time() - method_start:.3f}s with {len(ensemble_results)} results")
        
        # Combine results using different ensemble methods
        logger.info("Combining results using ensemble methods")
        
        # Method A: Reciprocal Rank Fusion
        ensemble_results = EnsembleRetrieverSystem.reciprocal_rank_fusion(results_lists)
        logger.debug(f"RRF produced {len(ensemble_results)} ranked documents")
        
        # Method B: Weighted Voting Ensemble
        weighted_results = EnsembleRetrieverSystem.weighted_voting_ensemble(results_lists, [0.2, 0.2, 0.2, 0.4])
        logger.debug(f"Weighted voting produced {len(weighted_results)} ranked documents")
        
        # Method C: Reranking Ensemble
        reranked_results = EnsembleRetrieverSystem.reranker_ensemble(query, results_lists, cross_encoder)
        logger.debug(f"Reranking produced {len(reranked_results)} ranked documents")
        
        # Final meta-ensemble: Select documents appearing in at least 2 of 3 result sets
        logger.info("Creating final meta-ensemble from top methods")
        
        doc_appearance_count = defaultdict(int)
        doc_relevance_scores = defaultdict(list)
        doc_id_to_doc = {}
        
        # Track appearances and scores in top-3 results of each strategy
        result_sets = [
            ("ReciprocalRankFusion", ensemble_results[:4]),
            ("WeightedVoting", weighted_results[:4]),
            ("CrossEncoderReranking", reranked_results[:4])
        ]
        
        # Process each ensemble method's top results
        for method_name, results in result_sets:
            logger.debug(f"Processing top results from {method_name}")
            for doc, score in results:
                doc_id = f"{doc.metadata['id']}_{doc.metadata['content_hash']}"
                doc_appearance_count[doc_id] += 1
                doc_relevance_scores[doc_id].append(score)
                doc_id_to_doc[doc_id] = doc
        
        # Create final filtered and sorted documents
        # Only include documents that appear in at least 2 of the ensemble methods
        final_ranked_docs = [
            (
                doc_id_to_doc[doc_id],
                np.mean(doc_relevance_scores[doc_id]),  # average_score
                doc_appearance_count[doc_id]  # appearance_count
            )
            for doc_id in doc_appearance_count if doc_appearance_count[doc_id] >= 2
        ]
        
        # Sort by average score descending
        final_results = sorted(final_ranked_docs, key=lambda x: x[1], reverse=True)[:5]
        
        # Log final results statistics
        logger.info(f"Ensemble retrieval completed in {time.time() - overall_start_time:.3f}s")
        logger.info(f"Returning {len(final_results)} documents that appeared in at least 2 ensemble methods")
        
        # Log details about top results
        for i, (doc, score, appearances) in enumerate(final_results):
            logger.debug(f"Result {i+1}: Doc {doc.metadata['id']}_{doc.metadata['content_hash']}, "
                        f"Score: {score:.4f}, Appearances: {appearances}")
        
        return final_results