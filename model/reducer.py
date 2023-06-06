from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing


class Reducer:
    def __init__(self, intermediate_components=50,
                 perplexity=30, num_components=2):
        self.intermediate_components = intermediate_components
        self.perplexity = perplexity
        self.num_components = num_components
        self.intermediate = intermediate_components is not None
        
        if self.intermediate:
            self.pre_reducer = PCA(n_components=self.intermediate_components)
        
        self.reducer = TSNE(n_components=self.num_components,
                            perplexity=self.perplexity,
                            metric='cosine'
                            )

    def reduce(self, embeddings):
        if self.intermediate:
            embeddings = self.pre_reducer.fit_transform(embeddings)
        
        return embeddings, self.reducer.fit_transform(embeddings)


class Scaler:
    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        
    def predict(self, embeddings):
        return self.scaler.fit_transform(embeddings)
