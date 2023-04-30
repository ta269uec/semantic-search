from sklearn.cluster import KMeans

def cluster_k_means(corpus, corpus_embeddings, num_clusters):
  clustering_model = KMeans(n_clusters=num_clusters)
  clustering_model.fit(corpus_embeddings)
  cluster_assignment = clustering_model.labels_
  clustered_sentences = [[] for i in range(num_clusters)]
  clustered_embeddings = [[] for i in range(num_clusters)]
  for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])
    clustered_embeddings[cluster_id].append(corpus_embeddings[sentence_id])
  return clustered_sentences, clustered_embeddings