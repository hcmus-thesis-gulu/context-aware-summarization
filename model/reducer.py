from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Reducer:
    def __init__(self, intermediate_components=50,
                 perplexity=30, num_components=2):
        self.intermediate_components = intermediate_components
        self.perplexity = perplexity
        self.num_components = num_components
        
        self.pre_reducer = PCA(n_components=self.intermediate_components)
        self.reducer = TSNE(n_components=self.num_components,
                            perplexity=self.perplexity,
                            metric='cosine'
                            )

    def reduce(self, embeddings):
        reduced_embeddings = self.pre_reducer.fit_transform(embeddings)
        return self.reducer.fit_transform(reduced_embeddings)
