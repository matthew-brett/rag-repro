import numpy as np

from scipy.sparse import linalg, csc_array

D = np.load('D.npy')
A = csc_array(D)
k = 415

o_vals = o_vectors = None

for i in range(100):
    rng = np.random.default_rng(1234)
    v0 = rng.random(A.shape[0])
    vals, vectors = linalg.eigsh(A, which='SM', v0=v0, k=k)
    if o_vals is None:
        o_vals, o_vectors = vals, vectors
    assert np.all(vals == o_vals)
    assert np.all(vectors == o_vectors)
