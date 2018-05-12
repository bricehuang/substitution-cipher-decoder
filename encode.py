import sys
import random
from decode import ALPHABET

def permutation():
    alphabet_copy = list(ALPHABET)
    random.shuffle(alphabet_copy)
    return {plain: cipher for plain, cipher in zip(ALPHABET, alphabet_copy)}

def encode(text):
    perm = permutation()
    return ''.join([perm[char] for char in text])

def retrieve_text(file):
    with open(file, 'r') as f:
        return f.readline().rstrip('\n\r')

def encode_file(plaintext_file, ciphertext_file):
    plaintext = retrieve_text(plaintext_file)
    ciphertext = encode(plaintext)
    f = open(ciphertext_file, 'w')
    f.write(ciphertext)
    f.close()

if __name__ == '__main__':
    plaintext_file = sys.argv[1]
    ciphertext_file = sys.argv[2] if len(sys.argv) >= 3 else plaintext_file
    encode_file(plaintext_file, ciphertext_file)
