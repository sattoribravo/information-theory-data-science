import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from itertools import combinations


def entropy(p):
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def joint_entropy(p_joint):
    p_joint = np.array(p_joint)
    p_joint = p_joint[p_joint > 0]
    return -np.sum(p_joint * np.log2(p_joint))


def conditional_entropy(p_xy, p_y):
    p_xy = np.array(p_xy)
    p_y = np.array(p_y)
    cond = p_xy / p_y[np.newaxis, :]
    cond = cond[p_xy > 0]
    return -np.sum(p_xy[p_xy > 0] * np.log2(cond))


def mutual_information(p_xy, p_x, p_y):
    p_xy = np.array(p_xy)
    p_x = np.array(p_x)
    p_y = np.array(p_y)
    mi = 0.0
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def kl_divergence_discrete(p, q):
    p = np.array(p)
    q = np.array(q)
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


def kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """KL divergence D_KL(N(mu1,sigma1^2) || N(mu2,sigma2^2)).
    Closed-form expression for univariate Gaussian distributions."""
    return (np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
            - 0.5)


def test_information_functions():
    print("Test Information Theory Functions")
    p = [0.25, 0.25, 0.25, 0.25]
    print("Entropy:", entropy(p))
    p_joint = [0.25, 0.25, 0.25, 0.25]
    p_x = [0.5, 0.5]
    p_y = [0.5, 0.5]
    print("Joint Entropy:", joint_entropy(p_joint))
    print("Conditional Entropy:", conditional_entropy(
        np.array(p_joint).reshape(2, 2), p_y))
    print("Mutual Information:", mutual_information(
        np.array(p_joint).reshape(2, 2), p_x, p_y))
    p1 = [0.5, 0.5]
    p2 = [0.9, 0.1]
    print("KL Divergence (discrete):", kl_divergence_discrete(p1, p2))
    print("KL Divergence (Gaussian):",
          kl_divergence_gaussian(0, 1, 1, 2))


def iris_analysis():
    print("\nIris Dataset Analysis")
    iris = load_iris()
    X = iris.data.astype(int)  # simple discretization
    feature_names = iris.feature_names
    pmfs = []
    for i in range(X.shape[1]):
        values, counts = np.unique(X[:, i], return_counts=True)
        pmf = counts / counts.sum()
        pmfs.append(pmf)
        plt.figure()
        plt.bar(values, pmf)
        plt.title(f"PMF of {feature_names[i]}")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(f"pmf_{feature_names[i].replace(' ', '_')}.png", dpi=300)
        plt.show()
    for i, pmf in enumerate(pmfs):
        print(f"Entropy of {feature_names[i]}:", entropy(pmf))
    num_features = X.shape[1]
    mi_matrix = np.zeros((num_features, num_features))
    for i, j in combinations(range(num_features), 2):
        pair_values, counts = np.unique(
            np.vstack([X[:, i], X[:, j]]).T, axis=0, return_counts=True)
        joint_pmf = counts / counts.sum()
        values_i = np.unique(X[:, i])
        values_j = np.unique(X[:, j])
        joint_table = np.zeros((len(values_i), len(values_j)))
        for idx, val in enumerate(pair_values):
            idx_i = np.where(values_i == val[0])[0][0]
            idx_j = np.where(values_j == val[1])[0][0]
            joint_table[idx_i, idx_j] = joint_pmf[idx]
        px = joint_table.sum(axis=1)
        py = joint_table.sum(axis=0)
        mi_matrix[i, j] = mutual_information(joint_table, px, py)
    max_idx = np.unravel_index(np.argmax(mi_matrix), mi_matrix.shape)
    print(f"Pair of features with highest Mutual Information: "
          f"{feature_names[max_idx[0]]} & {feature_names[max_idx[1]]}")
    print(f"Mutual Information value: {mi_matrix[max_idx]}")
    print("Conclusion: high mutual information indicates strong dependency.")


def gaussian_kl_analysis():
    print("\nGaussian KL Divergence Analysis")
    mus = np.linspace(0, 2, 5)
    sigmas = np.linspace(0.5, 2, 4)
    mu_ref = 0
    sigma_ref = 1
    for mu in mus:
        for sigma in sigmas:
            kl = kl_divergence_gaussian(mu_ref, sigma_ref, mu, sigma)
            print(f"KL(N(0,1) || N({mu:.2f},{sigma:.2f}^2)) = {kl:.4f}")


if __name__ == "__main__":
    test_information_functions()
    iris_analysis()
    gaussian_kl_analysis()
    input("Press Enter to exit...")
