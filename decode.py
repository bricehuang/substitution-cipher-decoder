import csv
import math
import numpy as np
import random
import sys

def extract_csv(filename):
    with open(filename, 'rb') as csvfile:
        return [row for row in csv.reader(csvfile)]
ALPHABET = extract_csv('./alphabet.csv')[0]
ALPHABET_SIZE = len(ALPHABET)
ALPHABET_INV = {ALPHABET[i]: i for i in xrange(ALPHABET_SIZE)}


def get_log_transition_probs():
    def _log_or_neg_inf(p):
        return math.log(p) if p > 0.0 else -20
    raw_transition_matrix = extract_csv('./letter_transition_matrix.csv')
    N = len(raw_transition_matrix)
    return np.array([
        [ _log_or_neg_inf(float(raw_transition_matrix[j][i])) for j in xrange(N)]
        for i in xrange(N)
    ])
LOG_TRANSITION_PROBS = get_log_transition_probs()

def get_transitions(text):
    transitions = [(text[i-1], text[i]) for i in xrange(1, len(text))]
    transition_counts = [[0 for j in xrange(ALPHABET_SIZE)] for i in xrange(ALPHABET_SIZE)]
    for begin, end in transitions:
        transition_counts[ALPHABET_INV[begin]][ALPHABET_INV[end]] += 1
    return np.array(transition_counts)

def compute_log_prob(perm, ciphertext_transitions):
    transition_counts = [
        [0 for i in xrange(ALPHABET_SIZE)] for j in xrange(ALPHABET_SIZE)
    ]
    for begin in xrange(ALPHABET_SIZE):
        for end in xrange(ALPHABET_SIZE):
            transition_counts[perm[begin]][perm[end]] = ciphertext_transitions[begin][end]
    transition_counts = np.array(transition_counts)
    return transition_counts.flatten().dot(LOG_TRANSITION_PROBS.flatten())

def initialize_perm_and_index_vector():
    perm = np.random.permutation(ALPHABET_SIZE)
    index_vector = np.concatenate((
        perm,
        perm+ALPHABET_SIZE,
        perm+2*ALPHABET_SIZE,
        perm+3*ALPHABET_SIZE,
        perm+4*ALPHABET_SIZE,
        perm+5*ALPHABET_SIZE,
        perm+6*ALPHABET_SIZE,
        perm+7*ALPHABET_SIZE,
        range(8*ALPHABET_SIZE, 8*ALPHABET_SIZE+16)
    ))
    return (perm, index_vector)

def gen_transposition():
    def rand_index():
        return random.randint(0, ALPHABET_SIZE-1)
    a = rand_index()
    b = rand_index()
    while (b == a):
        b = rand_index()
    return (a,b)

def apply_transposition(perm_and_index_vector, trans):
    perm, index_vector = perm_and_index_vector
    ct1, ct2 = trans
    pt1 = perm[ct1]
    pt2 = perm[ct2]
    perm[ct1] = pt2
    perm[ct2] = pt1
    for i in xrange(8):
        index_vector[ct1+i*ALPHABET_SIZE] = pt2+i*ALPHABET_SIZE
        index_vector[ct2+i*ALPHABET_SIZE] = pt1+i*ALPHABET_SIZE

def precompute_left(ciphertext_transitions):
    def _compute_entry(u,v):
        return np.concatenate((
            + ciphertext_transitions[v,:],
            + ciphertext_transitions[u,:],
            + ciphertext_transitions[:,v],
            + ciphertext_transitions[:,u],
            - ciphertext_transitions[u,:],
            - ciphertext_transitions[v,:],
            - ciphertext_transitions[:,u],
            - ciphertext_transitions[:,v],
            [
                + ciphertext_transitions[u,u],
                + ciphertext_transitions[u,v],
                + ciphertext_transitions[v,u],
                + ciphertext_transitions[v,v],
                + ciphertext_transitions[u,u],
                + ciphertext_transitions[u,v],
                + ciphertext_transitions[v,u],
                + ciphertext_transitions[v,v],
                - ciphertext_transitions[u,u],
                - ciphertext_transitions[u,u],
                - ciphertext_transitions[u,v],
                - ciphertext_transitions[u,v],
                - ciphertext_transitions[v,u],
                - ciphertext_transitions[v,u],
                - ciphertext_transitions[v,v],
                - ciphertext_transitions[v,v],
            ]
        ))
    return {
        (u,v): _compute_entry(u,v)
        for u in xrange(ALPHABET_SIZE) for v in xrange(ALPHABET_SIZE) if u != v
    }

def precompute_right():
    def _compute_entry(u,v):
        return np.concatenate((
            LOG_TRANSITION_PROBS[u,:],
            LOG_TRANSITION_PROBS[v,:],
            LOG_TRANSITION_PROBS[:,u],
            LOG_TRANSITION_PROBS[:,v],
            LOG_TRANSITION_PROBS[u,:],
            LOG_TRANSITION_PROBS[v,:],
            LOG_TRANSITION_PROBS[:,u],
            LOG_TRANSITION_PROBS[:,v],
            [
                LOG_TRANSITION_PROBS[u,u],
                LOG_TRANSITION_PROBS[u,v],
                LOG_TRANSITION_PROBS[v,u],
                LOG_TRANSITION_PROBS[v,v],
                LOG_TRANSITION_PROBS[v,v],
                LOG_TRANSITION_PROBS[v,u],
                LOG_TRANSITION_PROBS[u,v],
                LOG_TRANSITION_PROBS[u,u],
                LOG_TRANSITION_PROBS[u,v],
                LOG_TRANSITION_PROBS[v,u],
                LOG_TRANSITION_PROBS[u,u],
                LOG_TRANSITION_PROBS[v,v],
                LOG_TRANSITION_PROBS[u,u],
                LOG_TRANSITION_PROBS[v,v],
                LOG_TRANSITION_PROBS[u,v],
                LOG_TRANSITION_PROBS[v,u],
            ]
        ))
    return {
        (u,v): _compute_entry(u,v)
        for u in xrange(ALPHABET_SIZE) for v in xrange(ALPHABET_SIZE) if u != v
    }

def get_accept_log_prob(perm_and_index_vector, precomputes, trans):
    p, ind = perm_and_index_vector
    u,v = trans
    left_precomputes, right_precomputes = precomputes
    left = left_precomputes[(u,v)]
    right = right_precomputes[(p[u], p[v])]
    return np.dot(left, right[ind])

def mh_step(perm_and_index_vector, precomputes):
    trans = gen_transposition()
    accept_log_prob = min(get_accept_log_prob(perm_and_index_vector, precomputes, trans), 0)
    if (random.random() < math.exp(accept_log_prob)):
        # accept
        apply_transposition(perm_and_index_vector, trans)
        return True
    else:
        return False

CUTOFF_ON = True
STABLE_CUTOFF = 1000
ITERATIONS = 10000
def mh(ciphertext, ciphertext_transitions, ciphertext_precomputes):
    perm_and_index_vector = initialize_perm_and_index_vector()
    last_transition = 0
    for iteration in xrange(ITERATIONS):
        transition = mh_step(perm_and_index_vector, ciphertext_precomputes)
        if (transition):
            last_transition = iteration
        if (CUTOFF_ON and iteration - last_transition > STABLE_CUTOFF):
            break
    perm, _ = perm_and_index_vector
    entropy = -1. * compute_log_prob(perm, ciphertext_transitions) / len(ciphertext)
    return (perm, entropy)

METROPOLIS_HASTINGS_ATTEMPTS = 20
def decode(ciphertext, output_file_name):
    ciphertext_transitions = get_transitions(ciphertext)
    ciphertext_precomputes = (precompute_left(ciphertext_transitions), precompute_right())
    best_perm = None
    best_entropy = None
    for i in xrange(METROPOLIS_HASTINGS_ATTEMPTS):
        perm, entropy = mh(ciphertext, ciphertext_transitions, ciphertext_precomputes)
        if (best_entropy is None or entropy < best_entropy):
            best_perm = perm
            best_entropy = entropy
    def _decode_char(char):
        ciphertext_index = ALPHABET_INV[char]
        return ALPHABET[best_perm[ciphertext_index]]
    decoded = ''.join([_decode_char(char) for char in ciphertext])
    f = open(output_file_name, 'w')
    f.write(decoded)
    f.close()

def retrieve_text(file_name):
    with open(file_name, 'r') as f:
        return f.readline().rstrip('\n\r')

if __name__ == '__main__':
    input_file_name, output_file_name = sys.argv[1:3]
    ciphertext = retrieve_text(input_file_name)
    decode(ciphertext, output_file_name)
