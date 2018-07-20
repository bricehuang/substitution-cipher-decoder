# Substitution Cipher Decoder

This app decodes an arbitrary substitution cipher.  A substitution cipher is a cipher that replaces each plaintext character with a ciphertext character according to a fixed permutation of the alphabet.

This app uses a Markov Chain Monte Carlo algorithm and statistics of the English language to learn the ciphering permutation.  A full discussion of the methodology can be found [here](http://web.mit.edu/bmhuang/www/files/math/mcmc-decode.pdf).

## Getting Started

This app should work out of the box, on a computer with Python 2.7 and Numpy v1.1.10 installed.

## Usage

The main usage is
```
python decode.py [ciphertext_file] [output_file]
```
This writes to `output_file` the algorithm's decoding of `ciphertext_file`.  For example,
```
python decode.py sample_ciphertext/genesis.txt sample_decoded/genesis.txt
```
writes to `sample_decoded/genesis.txt` the algorithm's decoding of the ciphertext `sample_ciphertext/genesis.txt`.

## Encoding a Plaintext

The util script `encode.py` ciphers a plaintext by a uniformly random permutation of `'a'-'z',' ','.'`.  Usage is
```
python encode.py [plaintext_file] [ciphertext_file]
```
The plaintext should have the following properties:
* The text consists of characters from `'a'-'z',' ','.'`.
* A period (`'.'`) is always followed by a space (`' '`), and preceded by a letter (`'a'-'z'`).
* A space is always followed by a letter.
* The text is in English.

## Examples

Plaintext samples can be found in `sample_plaintext/`.  Ciphertexts of these plaintexts are in `sample_ciphertext/`, and decodings of these ciphertexts are in `sample_decoded/`.

## License

This project is licensed under the MIT License.
