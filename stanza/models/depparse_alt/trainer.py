import stanza.models.depparse.trainer
import numpy as np
import stanza.models.depparse_alt.alternatives as alternatives
from stanza.models.common import utils, loss

class Trainer(stanza.models.depparse.trainer.Trainer):
    def __init__(
            self, 
            n_parses=3, 
            kalm_shuffle=False, 
            automatic_n_parses=False,
            *args, 
            **kwargs
            ):
        super(Trainer, self).__init__(*args, **kwargs)
        self._n_trees = n_parses
        self._kalm_shuffle = kalm_shuffle
        self._automatic_n_parses = automatic_n_parses

    def predict(self, batch, unsort=True):
        inputs, orig_idx, word_orig_idx, sentlens, wordlens = unpack_batch(batch, self.use_cuda)
        word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel = inputs

        self.model.eval()
        batch_size = word.size(0)
        _, preds = self.model(word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens)
        # head_seqs is dimension (sentence, word)
        # so let's see what happens here
        # adj[:l, :l] is an adjacency matrix telling who is the head of who
        # preds[0][i] is the adjacency matrix for the i-th sentence
        # preds[1][i] is the depprel adjacency matrix for the i-th sentence

        # so basically: let's get the head-sequence for each sentence
        # let's get the 
        head_seqs = []
        deprel_seqs = []
        score_seqs = []
        # get the head graph and the deprel map for each sentence
        for i, (head, deps) in enumerate(zip(preds[0], preds[1])):
            head_seq = []
            deprel_seq = []
            score_seq = []
            edge_type = lambda edge: self.vocab['deprel'].unmap((deps[edge.v][edge.u],))[0]
            k_best = alternatives.GetKBest(
                    head, 
                    self._n_trees,
                    self._kalm_shuffle, 
                    edge_type,
                    self._automatic_n_parses,
                    )
            for j in range(sentlens[i] - 1):
                headc = []
                deprelc = []
                scorec = []
                for n_tree, (tree, score) in enumerate(k_best):
                    scorec.append(score)
                    best_in_edge = tree[j + 1]
                    source = best_in_edge.u - 1
                    headc.append(source + 1)
                    edge = self.vocab['deprel'].unmap((deps[j + 1][source + 1],))[0]
                    deprelc.append(edge)
                head_seq.append(headc)
                deprel_seq.append(deprelc)
                score_seq.append(scorec)

            head_seqs.append(head_seq)
            deprel_seqs.append(deprel_seq)
            score_seqs.append(score_seq)

        pred_tokens = [[[head_seqs[i][j], deprel_seqs[i][j], score_seqs[i][j]] for j in range(sentlens[i]-1)] for i in range(batch_size)]
        if unsort:
            pred_tokens = utils.unsort(pred_tokens, orig_idx)

        return pred_tokens

def unpack_batch(batch, use_cuda):
    """ Unpack a batch from the data loader. """
    if use_cuda:
        inputs = [b.cuda() if b is not None else None for b in batch[:11]]
    else:
        inputs = batch[:11]
    orig_idx = batch[11]
    word_orig_idx = batch[12]
    sentlens = batch[13]
    wordlens = batch[14]
    return inputs, orig_idx, word_orig_idx, sentlens, wordlens
