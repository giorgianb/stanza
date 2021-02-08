import itertools
import os

import stanza.models.depparse_alt.arbori as arbori
def GetKBest(weights, k, kalm_shuffle, edge_type, automatic_n_parses):
    V = set(range(len(weights)))
    E = set()
    scores = {}
    for i in range(len(weights)):
        for j in range(len(weights)):
            e = arbori.Edge(j, i)
            E.add(e)
            scores[e] = weights[i, j]
        
    return arbori.GetKBest(k, V, E, scores, kalm_shuffle, edge_type, automatic_n_parses)
