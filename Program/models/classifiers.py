from sklearn.cluster import SpectralClustering, DBSCAN, KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def clustering_object((clustering_method, parameters)):
    """
    Creates and returns specified clustering object.
    :return: object that can be used for clustering data
    """
    try:
        return {
            'spectral': _get_spectral(parameters),
            'dbscan': _get_dbscan(parameters),
            'kmeans': _get_kmeans(parameters)
        }[clustering_method]
    except KeyError:
        raise ValueError('Provided clustering method name \'' + str(clustering_method) + '\' not recognized.')


def classifier_object((classification_method, parameters)):
    """
    Creates and returns specified classifier object.
    :return: classifier object
    """
    try:
        return {
            'svm': _get_svm(parameters),
            'rf': _get_rf(parameters),
            'knn': _get_knn(parameters),
        }[classification_method]
    except KeyError:
        raise ValueError('Provided classifier name \'' + str(classification_method) + '\' not recognized.')


def _get_spectral(parameters):
    if parameters is None:
        parameters = {
            'n_clusters': 2,
            'affinity': 'nearest_neighbors'
        }
    return SpectralClustering(**parameters)


def _get_dbscan(parameters):
    if parameters is None:
        parameters = {
        }
    return DBSCAN(**parameters)


def _get_kmeans(parameters):
    if parameters is None:
        parameters = {
            'n_clusters': 2,
            'n_jobs': -1
        }
    return KMeans(**parameters)


def _get_svm(parameters):
    if parameters is None:
        parameters = {
            'C': 8,
            'kernel': 'rbf',
            'gamma': 0.5
        }
    return svm.SVC(**parameters)


def _get_rf(parameters):
    if parameters is None:
        parameters = {
            'n_estimators': 100,
        }
    return RandomForestClassifier(**parameters)


def _get_knn(parameters):
    if parameters is None:
        parameters = {
            'n_neighbors': 5
        }
    return KNeighborsClassifier(**parameters)
