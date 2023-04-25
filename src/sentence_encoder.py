from sentence_transformers import SentenceTransformer
     
class SentenceEncoder():
    def __init__(self, model='sentence-transformers/all-MiniLM-L12-v2', device='cuda'):
        self.model = SentenceTransformer(model, device)
        pass
    
    def encode(self, sentences):
        return self.model.encode(sentences)