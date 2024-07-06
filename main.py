import torch

from decoder_lm.tokenizer import WordGPTTokenizer

if __name__ == "__main__":
    tokens = ["hello", "world", "@thiomajid", "is", "there", "<EOS>"]
    # vocabulary: dict[str, int] = {}
    # for idx, token in enumerate(tokens):
    #     vocabulary[token] = idx
    vocabulary = {token: idx for token, idx in zip(tokens, range(len(tokens)))}

    print(vocabulary)
    tokenizer = WordGPTTokenizer.from_dict(vocabulary)
    print(tokenizer.decode(torch.tensor([0, 3])))
