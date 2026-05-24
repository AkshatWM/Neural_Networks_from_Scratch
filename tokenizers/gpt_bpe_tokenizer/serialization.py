from pathlib import Path

def save_tokenizer(tokenizer, file_prefix):

    file_prefix = Path(file_prefix)

    model_file = file_prefix.with_suffix(".model")

    with open(model_file, "w", encoding="utf-8") as f:

        f.write("bpe v1\n")

        # regex pattern
        f.write(f"{tokenizer.pattern}\n")

        # merges ordered by token id
        for pair, idx in sorted(
            tokenizer.merges.items(),
            key=lambda x: x[1]
        ):

            p0, p1 = pair

            f.write(f"{p0} {p1}\n")


    vocab_file = file_prefix.with_suffix(".vocab")

    inverted_merges = {
        idx: pair
        for pair, idx in tokenizer.merges.items()
    }

    with open(vocab_file, "w", encoding="utf-8") as f:

        for idx, token_bytes in tokenizer.vocab.items():

            s = render_token(token_bytes)

            if idx in inverted_merges:

                p0, p1 = inverted_merges[idx]

                s0 = render_token(tokenizer.vocab[p0])
                s1 = render_token(tokenizer.vocab[p1])

                f.write(
                    f"[{s0}][{s1}] -> [{s}] {idx}\n"
                )

            else:
                f.write(f"[{s}] {idx}\n")


def load_tokenizer(tokenizer_cls, model_file):

    tokenizer = tokenizer_cls()

    merges = {}

    with open(model_file, "r", encoding="utf-8") as f:

        # version
        version = f.readline().strip()

        if version != "minbpe v1":
            raise ValueError("Unknown tokenizer version")

        # regex pattern
        tokenizer.pattern = f.readline().strip()

        # rebuild merges
        idx = 256

        for line in f:

            p0, p1 = map(int, line.split())

            merges[(p0, p1)] = idx

            idx += 1

    tokenizer.merges = merges

    # rebuild vocab
    tokenizer.vocab = build_vocab(merges)

    return tokenizer


def build_vocab(merges):

    vocab = {idx: bytes([idx]) for idx in range(256)}

    for (p0, p1), idx in sorted(
        merges.items(),
        key=lambda x: x[1]
    ):

        vocab[idx] = vocab[p0] + vocab[p1]

    return vocab


def render_token(token_bytes):

    s = token_bytes.decode(
        "utf-8",
        errors="replace"
    )

    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")

    return s