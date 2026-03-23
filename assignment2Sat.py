import numpy as np
import pandas as pd


def load_dataset(path, class_col=-1):
    df = pd.read_csv(path)
    if isinstance(class_col, int):
        y = df.iloc[:, class_col].values
        X = df.drop(df.columns[class_col], axis=1).values
    else:
        y = df[class_col].values
        X = df.drop(columns=class_col).values
    return X, y


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def split_train_test_by_class(X, y, train_ratio=0.5, random_state=42):
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        train_idx = idx[:n_train]
        test_idx = idx[n_train:]
        X_train_list.append(X[train_idx])
        X_test_list.append(X[test_idx])
        y_train_list.append(y[train_idx])
        y_test_list.append(y[test_idx])
    X_train = np.vstack(X_train_list)
    X_test = np.vstack(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    return X_train, X_test, y_train, y_test


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.vars = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        for idx, c in enumerate(self.classes):
            Xc = X[y == c]
            self.means[idx] = Xc.mean(axis=0)
            self.vars[idx] = Xc.var(axis=0) + 1e-9
            self.priors[idx] = Xc.shape[0] / X.shape[0]

    def _gaussian_log_pdf(self, class_idx, X):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * var) + (X - mean) ** 2 / var, axis=1)

    def predict(self, X):
        log_posteriors = []
        for idx in range(len(self.classes)):
            log_prior = np.log(self.priors[idx])
            log_likelihood = self._gaussian_log_pdf(idx, X)
            log_posteriors.append(log_prior + log_likelihood)
        log_posteriors = np.vstack(log_posteriors).T
        return self.classes[np.argmax(log_posteriors, axis=1)]


class NaiveBayesHistogram:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.bin_edges = [[None] * n_features for _ in range(n_classes)]
        self.bin_probs = [[None] * n_features for _ in range(n_classes)]
        self.priors = np.zeros(n_classes)
        for ci, c in enumerate(self.classes):
            Xc = X[y == c]
            self.priors[ci] = Xc.shape[0] / X.shape[0]
            for fi in range(n_features):
                counts, edges = np.histogram(
                    Xc[:, fi], bins=self.n_bins, density=False)
                counts = counts.astype(float) + 1.0  # Laplace smoothing
                probs = counts / counts.sum()
                self.bin_edges[ci][fi] = edges
                self.bin_probs[ci][fi] = probs

    def predict(self, X):
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))
        for ci in range(n_classes):
            log_posteriors[:, ci] = np.log(self.priors[ci])
            for fi in range(n_features):
                edges = self.bin_edges[ci][fi]
                probs = self.bin_probs[ci][fi]
                idx = np.searchsorted(edges, X[:, fi], side='right') - 1
                idx = np.clip(idx, 0, len(probs) - 1)
                log_posteriors[:, ci] += np.log(probs[idx])
        return self.classes[np.argmax(log_posteriors, axis=1)]


class BayesMultivariateGaussian:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        self.means = np.zeros((n_classes, n_features))
        self.cov_inv = []
        self.cov_log_det = np.zeros(n_classes)
        self.priors = np.zeros(n_classes)
        for ci, c in enumerate(self.classes):
            Xc = X[y == c]
            self.priors[ci] = Xc.shape[0] / X.shape[0]
            mean = Xc.mean(axis=0)
            cov = np.cov(Xc, rowvar=False) + 1e-6 * np.eye(n_features)
            sign, logdet = np.linalg.slogdet(cov)
            self.means[ci] = mean
            self.cov_inv.append(np.linalg.inv(cov))
            self.cov_log_det[ci] = logdet

    def _log_multivariate_gaussian(self, ci, X):
        diff = X - self.means[ci]
        quad = np.einsum('ij,jk,ik->i', diff, self.cov_inv[ci], diff)
        d = X.shape[1]
        return -0.5 * (d * np.log(2.0 * np.pi) + self.cov_log_det[ci] + quad)

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))
        for ci in range(n_classes):
            log_posteriors[:, ci] = (
                np.log(self.priors[ci])
                + self._log_multivariate_gaussian(ci, X))
        return self.classes[np.argmax(log_posteriors, axis=1)]


if __name__ == '__main__':
    X, y = load_dataset('data.csv', class_col=-1)
    X_train, X_test, y_train, y_test = split_train_test_by_class(
        X, y, train_ratio=0.5, random_state=42)

    bin_values = [5, 10, 20]

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)
    print('Gaussian NB accuracy:', accuracy_score(y_test, gnb.predict(X_test)))

    for n_bins in bin_values:
        nb = NaiveBayesHistogram(n_bins=n_bins)
        nb.fit(X_train, y_train)
        print(f'Histogram NB bins={n_bins} accuracy:',
              accuracy_score(y_test, nb.predict(X_test)))

    mgb = BayesMultivariateGaussian()
    mgb.fit(X_train, y_train)
    print('Multivariate Gaussian Bayes accuracy:',
          accuracy_score(y_test, mgb.predict(X_test)))
