from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def Visualization(HiddenVector,ExpectedOutputs, ModelName):
    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(HiddenVector)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    #Run T-SNE on the PCA features.
    tsne = TSNE(n_components= 2, verbose = 1)
    tsne_results = tsne.fit_transform(HiddenVector)
    y_test_cat = np_utils.to_categorical(ExpectedOutputs ,num_classes = 3)
    color_map = np.argmax(y_test_cat, axis=1)
    plt.figure(figsize=(5,5))
    for cl in range(3):
        indices = np.where(color_map==cl)
        indices = indices[0]
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
    plt.title(ModelName)
    plt.legend()
    plt.show()