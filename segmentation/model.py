from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_clustering(X, n_clusters, random_state):
    X_scaled = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=0.99, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_pca)

    return labels, X_pca, km, pca
