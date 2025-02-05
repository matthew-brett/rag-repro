import numpy as np
from scipy.sparse import linalg

from skimage import segmentation, data
from skimage import graph
ggc = graph._graph_cut
from numpy.testing import assert_array_equal

thresh=1e-3
num_cuts=10
max_edge=1.0

img = data.coffee()
labels1 = segmentation.slic(img, compactness=30, n_segments=400, start_label=0)
g = graph.rag_mean_color(img, labels1, mode='similarity')
for node in g.nodes():
    g.add_edge(node, node, weight=max_edge)
d, w = graph._ncut.DW_matrices(g)
m = w.shape[0]

results = [None] * 4


# Since d is diagonal, we can directly operate on its data
# the inverse of the square root
d2 = d.copy()
d2.data = np.reciprocal(np.sqrt(d2.data, out=d2.data), out=d2.data)

# Refer Shi & Malik 2001, Equation 7, Page 891
A = d2 @ (d - w) @ d2

rng = np.random.default_rng(seed=1234)
v0 = rng.random(A.shape[0])
k = min(100, m - 2)

o_vals, o_vectors = linalg.eigsh(A, which='SM', v0=v0, k=k)

for i in range(len(results)):

    rag = g.copy()

    # Initialize the vector to ensure reproducibility.
    vals, vectors = linalg.eigsh(A, which='SM', v0=v0, k=min(100, m - 2))

    assert np.all(o_vals == vals)
    assert np.all(o_vectors == vectors)

    # Pick second smallest eigenvector.
    # Refer Shi & Malik 2001, Section 3.2.3, Page 893
    vals, vectors = np.real(vals), np.real(vectors)
    index2 = ggc._ncut_cy.argmin2(vals)
    ev = vectors[:, index2]

    cut_mask, mcut = ggc.get_min_ncut(ev, d, w, num_cuts)
    if mcut >= thresh:
        print('Threshold not met')
        ggc._label_all(rag, 'ncut label')
    # Sub divide and perform N-cut again
    # Refer Shi & Malik 2001, Section 3.2.5, Page 893
    sub1, sub2 = ggc.partition_by_cut(cut_mask, rag)

    ggc._ncut_relabel(sub1, thresh, num_cuts, rng)
    ggc._ncut_relabel(sub2, thresh, num_cuts, rng)

    map_array = np.zeros(labels1.max() + 1, dtype=labels1.dtype)

    # Mapping from old labels to new
    for node, n_d in rag.nodes(data=True):
        map_array[n_d['labels']] = n_d['ncut label']

    results[i] = map_array


for i in range(len(results) - 1):
    assert_array_equal(results[i], results[i + 1])
