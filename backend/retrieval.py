from typing import List, Dict, Any
from embeddings import OllamaEmbeddings
from vector_store import FAISSVectorStore


class RetrievalPipeline:
    """
    Handles query processing, embedding, and context retrieval.
    Orchestrates the semantic search process.
    """
    
    def __init__(self, vector_store: FAISSVectorStore, embeddings: OllamaEmbeddings):
        """
        Initialize retrieval pipeline.
        
        Args:
            vector_store: FAISS vector store instance
            embeddings: Ollama embeddings instance
        """
        self.vector_store = vector_store
        self.embeddings = embeddings
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Main retrieval method: converts query to embedding and finds relevant chunks.
        
        Args:
            query: User's legal question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        try:
            # Step 1: Embed the query
            print(f"Embedding query: '{query[:100]}...'")
            query_embedding = self.embeddings.embed_text(query)
            
            if not query_embedding:
                return {
                    'success': False,
                    'error': 'Failed to generate query embedding',
                    'results': []
                }
            
            # Step 2: Search vector store
            print(f"Searching for top {top_k} relevant chunks...")
            results = self.vector_store.search(query_embedding, top_k=top_k)
            
            if not results:
                return {
                    'success': True,
                    'message': 'No relevant documents found',
                    'results': []
                }
            
            # Step 3: Format results for generation
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'rank': result['rank'],
                    'similarity': result['similarity'],
                    'content': result['metadata'].get('content', ''),
                    'case_name': result['metadata'].get('case_name', 'Unknown'),
                    'section_type': result['metadata'].get('section_type', 'general'),
                    'paragraph_range': result['metadata'].get('paragraph_range', 'N/A')
                })
            
            print(f"Retrieved {len(formatted_results)} relevant chunks")
            
            return {
                'success': True,
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Retrieval error: {str(e)}',
                'results': []
            }
    
    def format_context_for_generation(self, retrieval_results: Dict[str, Any]) -> str:
        """
        Convert retrieved chunks into a formatted context string for LLM.
        
        Args:
            retrieval_results: Output from retrieve_context()
            
        Returns:
            Formatted context string
        """
        if not retrieval_results.get('success') or not retrieval_results.get('results'):
            return ""
        
        context_parts = []
        
        for result in retrieval_results['results']:
            context_parts.append(
                f"[Source: {result['case_name']} - {result['section_type']} - Para {result['paragraph_range']}]\n"
                f"{result['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_all_chunks_for_summary(self) -> Dict[str, Any]:
        """
        Retrieve all chunks from vector store for case summary generation.
        
        Returns:
            Dictionary with all chunks organized by section type
        """
        try:
            if not self.vector_store or not self.vector_store.metadata:
                return {
                    'success': False,
                    'error': 'No documents available',
                    'chunks': []
                }
            
            # Get all metadata (which contains content and section info)
            all_chunks = []
            for metadata in self.vector_store.metadata:
                all_chunks.append({
                    'content': metadata.get('content', ''),
                    'case_name': metadata.get('case_name', 'Unknown'),
                    'section_type': metadata.get('section_type', 'general'),
                    'paragraph_range': metadata.get('paragraph_range', 'N/A')
                })
            
            # Organize by section type
            organized = {}
            for chunk in all_chunks:
                section = chunk['section_type']
                if section not in organized:
                    organized[section] = []
                organized[section].append(chunk)
            
            return {
                'success': True,
                'chunks': all_chunks,
                'organized': organized,
                'total_chunks': len(all_chunks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error retrieving chunks: {str(e)}',
                'chunks': []
            }