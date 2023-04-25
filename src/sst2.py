from datasets import list_datasets, load_dataset

def get_reviews_sst2(split="train"):
  dataset = load_dataset("sst2", split=split)
  neg, pos = dataset.filter(lambda x:x['label']==0), dataset.filter(lambda x:x['label']==1)
  negative_sentences, positive_sentences = [neg[i]['sentence'] for i in range(len(neg))], [pos[i]['sentence'] for i in range(len(pos))]
  return positive_sentences, negative_sentences