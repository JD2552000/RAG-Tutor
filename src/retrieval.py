#This file retreives the most meaningful chunks from the vector_db and then pass as to LLM as augmented prompt

from flashrank import Ranker, RerankRequest
from src.database import VectorDatabase

class AdvancedRetriever:
    def __init__(self):
        self.db = VectorDatabase()
        self.ranker = Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp')
    
    def search(self, query: str, top_k=10):
        #Get initial results from the Qdrant db
        initial_docs = self.db.vector_store.similarity_search(query, k=top_k)

        #Format for Flashrank
        passages=[
            {'id':i, 'text':doc.page_content, 'meta'=doc.metadata}
            for i, doc in enumerate(initial_docs)
        ]

        #Re-rank the results
        rerank_request = RerankRequest(query=query, passages=passages)
        results= self.ranker.rerank(rerank_request)

        #We will currently only return the top 3 chunks that are most relevant
        return [r['text'] for r in results[:3]]