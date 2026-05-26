#byte-level BPE tokenizer
import regex as re
from .basic import BasicTokenizer
from .base import get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):

    def __init__(self):
        super().__init__()
        self.merges = {}
        self.pattern = GPT4_SPLIT_PATTERN

    def train(self, text, vocab_size, verbose=None):

        text_chunks = re.findall(self.pattern, text)
        
        text_bytes = [chunk.encode('utf-8') for chunk in text_chunks]
        ids = [list(b) for b in text_bytes]

        num_merges = vocab_size - 256
        merges = {} 
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            if len(stats) == 0:
                break
            
            pair = max(stats, key = stats.get)
            idx = 256+i
            
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def decode(self, ids):
        #Given a list of integers we convert it to text
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    def encode_chunk(self, chunk_bytes):

        ids = list(chunk_bytes)
        
        while len(ids) >= 2:
            pairs = get_stats(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
        
    def encode(self, text):
        #Given a string we convert it to list of integers based on what the tokenizer has learned
        text_chunks = re.findall(GPT4_SPLIT_PATTERN, text)
        text_bytes = [chunk.encode('utf-8') for chunk in text_chunks]
        ids = []
        
        for chunk_bytes in text_bytes:
            chunk_ids = self.encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids