"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net import Net
from utils import log_sum_exp


class BiLstm_TorchCrf(Net):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(BiLstm_TorchCrf, self).__init__(params)

        # the embedding takes as input the vocab_size and the embedding_dim
        # self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.bilstm = nn.LSTM(params.embedding_dim,
                            params.lstm_hidden_dim,
                            batch_first=True,
                            bidirectional=True,
                            num_layers=2
                            )

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Linear(params.lstm_hidden_dim*2, params.number_of_tags) # TO DO add and rename this parameter
        # self.crf = torchcrf.CRF(num_tags=9, batch_first=True)  # TODO softcode
        # print("="*45, "BILSTM CRF IN USE", "="*45 )

        self.transitions = nn.Parameter(torch.randn(params.number_of_tags, params.number_of_tags))  # TODO check random seed
        self.start = nn.Parameter(torch.rand(params.number_of_tags))
        self.end = nn.Parameter(torch.rand(params.number_of_tags))

        self.batch_size = params.batch_size
        self.n_tags = params.number_of_tags

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)

    def _bilstm_emissions_prob(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        # -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim
        s = self.embedding(s)

        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        s, _ = self.bilstm(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        s = s.view(-1,  s.shape[2])

        # apply the fully connected layer and obtain the output (before softmax) for each token
        # dim: batch_size*seq_len x num_tags
        s = self.fc(s)

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        s = s.view(self.batch_size, -1, self.n_tags)
        s = F.log_softmax(s, dim=2)  # dim=2 for torch-crf cause of additional batch size dim

        # dim: batch_size x seq_len x num_tags
        return s

    def score_sequence(self, emission_logits, tags, mask):

        """comupte denominator"""
        # TODO consider reshaping / transposing ? # lines 273 to 275 in Allemnlp code
        # TODO consider START AND END tags
        # -- Note in that case, pass extreme values for START and END tags to Transition matrix
        # and add both tags (with build_vocab hyperparameters) to the tags.txt, finally set indices in DataLoader

        batch_size, seq_len, n_tags = emission_logits.data.shape

        score = torch.zeros(1)
        for t in range(seq_len - 1):

            # compute unary logs or emission score
            #  emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            unary_factor = emission_logits[t]

            # compute transition score
            # transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            pairwise_factor = self.transtitions[tags[t], tags[t+1]]

            score += unary_factor * mask[t] + pairwise_factor * mask[t+1]

        return score

    def the_forward_algorithm(self, emission_logits, tags, mask):
        """"Compute the partition function, a.k.a normalization constant using forward algorithm
        see Jufrasky, and H.Martin, Speech and Language Processing,
        appendex A, Section A.3, https://web.stanford.edu/~jurafsky/slp3/A.pdf
        :returns the forward probability """

        batch_size, seq_len, n_tags = emission_logits.data.shape

        forward_prob = emission_logits[0]

        for t in range(1, seq_len):
            # TODO comment code

            # broadcast the three factors along different axis adequately
            unary_factor = emission_logits[t].view(batch_size, 1, n_tags)
            pairwise_factor = self.transitions.view(1, n_tags, n_tags )
            forward_prob_broadcast = forward_prob.view(batch_size, n_tags, 1)

            trellis_cell = forward_prob_broadcast + pairwise_factor + unary_factor
            # compute log_sum_exp for valid positions, otherwise pass previous log probability
            forward_prob = torch.where(mask[t].view(batch_size, 1),
                                       log_sum_exp(trellis_cell),
                                       forward_prob
                                       )

        return forward_prob

    def viterbi_decoding(self, sentence_len, n_tags):
        """"Viterbi decoding algorithm for finding the optimal tag sequence
        :param
        sentence_len : observation of len T
        n_tags : states of len N

        :returns best_tag_sequence, best_score

        pseudocode notes:
        lattice[N, T] a sequence probability matrix
        for s (or tag) in N:
            lattice[s,1] = start_prob * observation_likelihood (or emission prob)
            backpointer[s, 1] = 0


        for w (or word) in range[2, T]:
            for s in N:
                lattice[s, t] = Max (lattice[s_previous, t-1] * observation_likelihood * pairwise_potential)
                backpointer[s, t] = Argmax (lattice[s_previous, t-1] * observation_likelihood * pairwise_potential)
                # line above is for HMM Need to check changes to linear chainCRF
                # according to Jufrasky book Ch.8 pp. 20 see eq. 8.33 for more details
                # v_t(s) = max_{1}^{N} v_{t-1}(i) * sum_{1}^{K} (w_kf_k(y_{t-1}, y_t, X, t)

        best_score = Max lattice[s, t]
        best_path = Argamx viterbi[s, t]

        """
        raise NotImplementedError



    def NLL_loss(self):
        """returns Negative log likelihood"""
        raise NotImplementedError

    def forward(self, s):
        """returns best tags list """

        return self._bilstm_emissions_prob(s)

    def forward_alg(self):
        """dynamic programming to compute alphas ?? """
        raise NotImplementedError

    def train_Bilstm_Crf(self):
        """have to figure out how the output of the forward method is used:
        all implementations does not call it as usual in pytorch training loop
        but seems to compute directly NLL by loss = model.NLL().
        if the forward is performed under the hood, which is very likely,
        why the output is not used to compute the loss?
        TBS, one should consider that the loss is backprobagated though these parameters

        https://twitter.com/karpathy/status/904808674845007872
        exlanation of the way model is used in training.
        Avoid passing true labels before forward pass
        """
        raise NotImplementedError





