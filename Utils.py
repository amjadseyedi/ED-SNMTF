import numpy as np
from scipy.sparse.linalg import svds

def orthNNLS(M, U, Mn=None):
    """
    Solves the following optimization problem:
    min_{norm2v >= 0, V >= 0 and VV^T = D} ||M - U * V||_F^2

    Parameters:
        M (numpy.ndarray): Matrix M of size (m, n).
        U (numpy.ndarray): Matrix U of size (m, r).
        Mn (numpy.ndarray, optional): Normalized columns of M. If None, it will be computed.

    Returns:
        V (numpy.ndarray): The matrix V of size (r, n) that approximates M.
        norm2v (numpy.ndarray): The squared norms of the columns of V.
    see F. Pompili, N. Gillis, P.-A. Absil and F. Glineur, "Two Algorithms for
    Orthogonal Nonnegative Matrix Factorization with Application to
    Clustering", Neurocomputing 141, pp. 15-25, 2014.

    """

    if Mn is None:
        # Normalize columns of M
        norm2m = np.sqrt(np.sum(M ** 2, axis=0))  # norm2m is the L2 norm of each column of M
        Mn = M * (1 / (norm2m + 1e-16))  # Avoid division by zero

    m, n = Mn.shape
    m_, r = U.shape

    # Normalize columns of U
    norm2u = np.sqrt(np.sum(U ** 2, axis=0))  # norm2u is the L2 norm of each column of U
    Un = U * (1 / (norm2u + 1e-16))  # Avoid division by zero

    # Calculate the matrix A, which is the angles between columns of M and U
    A = np.dot(Mn.T, Un)  # A is (n, r), matrix of angles

    # Find the index of the maximum value in each row of A (best column of U to approximate each column of M)
    b = np.argmax(A, axis=1)  # Indices of the best matching column in U

    # Initialize V with zeros
    V = np.zeros((r, n))

    # Assign the optimal weights to V(b(i), i)
    for i in range(n):
        V[b[i], i] = np.dot(M[:, i], U[:, b[i]]) / norm2u[b[i]] ** 2

    return V

def update_orth_basis(V, v):
    """
    Updates the orthonormal basis V by adding a new vector v while ensuring orthogonality.

    Parameters:
        V (numpy.ndarray): Current orthonormal basis (m x k) where columns are basis vectors.
        v (numpy.ndarray): New vector to be added (m,).

    Returns:
        numpy.ndarray: Updated orthonormal basis including v.
    """
    if V.size == 0:
        # If V is empty, normalize v and set it as the first basis vector
        V = v / np.linalg.norm(v).reshape(1, -1)
        V=V.T
    else:
        # Project v onto the orthogonal complement of V
        v = v - V @ (V.T @ v)
        # Normalize v
        v = v / np.linalg.norm(v)
        # Append to the basis
        V = np.column_stack((V, v))

    return V

def SVCA(X,r,p,options=None):
    """
        Smoothed Vertex Component Analysis(SVCA)

        Heuristic to solve the following problem:
        Given a matrix X, find a matrix W such that X~=WH for some H>=0,
        under the assumption that each column of W has p columns of X close
        to it (called the p proximal latent points).

        Parameters:
            X (numpy.ndarray): Input data matrix of size (m, n).
            r (int): Number of columns of W.
            p (int): Number of proximal latent points.
            options (dict, optional):
                - 'lra' (int):
                    1 uses a low-rank approximation (LRA) of X in the selection step,
                    0 (default) does not.
                - 'average' (int):
                    1 uses the mean for aggregation,
                    0 (default) uses the median.

        Returns:
            W (numpy.ndarray): The matrix such that X ≈ WH.
            K (numpy.ndarray): Indices of the selected data points (one column per iteration).

            This code is based on the paper Smoothed Separable Nonnegative Matrix Factorization
            by N. Nadisic, N. Gillis, and C. Kervazo
            https://arxiv.org/abs/2110.05528
        """
    if options is None:
        options = {}

        # Approximation de faible rang (LRA) de la matrice d'entrée
        # Par défaut, il n'y a pas d'approximation de faible rang
        # Set default options if not provided
    X = X.astype('float')
    if 'lra' not in options:
        options['lra'] = 0

        # Calcul des vecteurs singuliers
    U, S, Vt = svds(X, k=r)  # U contient les premiers r vecteurs singuliers de X

    # On peut utiliser des algorithmes plus rapides ici, comme mentionné dans le code MATLAB

    if options['lra'] == 1:
        X = np.dot(S, Vt)  # Remplace X par son approximation de faible rang

    # Agrégation par moyenne ou médiane (par défaut : médiane)
    if 'average' not in options:
        options['average'] = 0  # Médiane par défaut

    # Projection (I - VV^T) sur le complément orthogonal des colonnes de W
    V = np.empty((X.shape[0], 0))  # Matrice vide pour commencer les itérations de SVCA

    W = np.zeros((X.shape[0], r))  # Matrice W de la taille m * r
    K = np.zeros((r, p), dtype=int)  # Indices des points de données sélectionnés

    for k in range(r):
        # Direction aléatoire dans la colonne de U
        # diru = np.dot(U, np.random.randn(r))
        diru = U[:, k]

        # Projection de la direction aléatoire pour être orthogonale aux colonnes extraites de W
        if k >= 1:
            diru = diru - np.dot(V, np.dot(V.T, diru))

        # Produit scalaire avec la matrice de données
        u = np.dot(diru.T, X)

        # Trier les entrées et sélectionner la direction maximisant |u|
        b = np.argsort(u)

        # Vérifier la condition de médiane
        if np.abs(np.median(u[b[:p]])) < np.abs(np.median(u[b[-p:]])):
            b = b[::-1]  # Inverser si nécessaire

        # Sélectionner les indices correspondant aux plus grandes valeurs de u
        K[k, :] = b[:p]

        # Calcul de la "vertex"
        if p == 1:
            W[:, k] = X[:, K[k, :]]
        else:
            if options['average'] == 1:
                W[:, k] = np.mean(X[:, K[k, :]], axis=1)
            else:
                W[:, k] = np.median(X[:, K[k, :]], axis=1)

        # Mise à jour du projecteur
        V = update_orth_basis(V, W[:, k])

    if options['lra'] == 1:
        W = np.dot(U, W)  # Si l'approximation de faible rang est activée, on multiplie par U

    return W