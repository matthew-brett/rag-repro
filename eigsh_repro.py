import numpy as np
from scipy.sparse import linalg

from skimage import segmentation, data
from skimage import graph
import skimage.graph._graph_cut as ggc


img = data.coffee()
labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=0)
g = graph.rag_mean_color(img, labels1, mode='similarity')
for node in g.nodes():
    g.add_edge(node, node, weight=1.0)
d, w = ggc._ncut.DW_matrices(g)
m = w.shape[0]

d2 = d.copy()
d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

# Refer Shi & Malik 2001, Equation 7, Page 891
A = d2 @ (d - w) @ d2

rng = np.random.default_rng(seed=1234)

v0 = rng.random(A.shape[0])

k = min(100, m - 2)

o_vals, o_vectors = linalg.eigsh(A, which='SM', v0=v0, k=k)

for i in range(10):

    # Initialize the vector to ensure reproducibility.
    vals, vectors = linalg.eigsh(A, which='SM', v0=v0, k=k)

    assert np.all(o_vals == vals)  # Fails.
