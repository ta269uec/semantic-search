from sentence_transformers import SentenceTransformer, util
     
class SentenceEncoder():
    def __init__(self, model='sentence-transformers/all-MiniLM-L12-v2', device='cuda'):
        self.model = SentenceTransformer(model, device)
        pass
    
    def encode(self, sentences, convert_to_tensor=True):
        return self.model.encode(sentences, convert_to_tensor=convert_to_tensor)