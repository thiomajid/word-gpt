import copy
import json
from pathlib import Path

import torch


class WordGPTTokenizer:
    """
    A custom word-based tokenizer whose behavior is inspired from
    `transformers.Tokenizer`.
    """

    _itoa: dict[int, str]
    _atoi: dict[str, int]

    @staticmethod
    def from_dict(vocabulary: dict[str, int]) -> "WordGPTTokenizer":
        """
        Creates a tokenizer with a user provided vocabulary dictionnary object

        Parameters
        ----------
        vocabulary : dict[str, int]
            The vocabulary holding a mapping of token => token_id

        Returns
        -------
        WordGPTTokenizer

        Raises
        ------
        ValueError
            If a None object has been provided as vocabulary
        """
        if vocabulary is None:
            raise ValueError("The tokenizer's vocabulary can not be set to None.")

        tokenizer = WordGPTTokenizer.__new__(WordGPTTokenizer)
        tokenizer._atoi = copy.deepcopy(vocabulary)
        tokenizer._itoa = {v: k for k, v in tokenizer._atoi.items()}

        return tokenizer

    @staticmethod
    def from_config_file(vocab_path: Path | str) -> "WordGPTTokenizer":
        """
        Creates a tokenizer and loads its configuration file from the provided path.

        Parameters
        ----------
        config_path : Path | str
            The location of the tokenizer's vocabulary on the local machine.
            It must be a json file.

        Returns
        -------
        Tokenizer

        Raises
        ------
        ValueError
            If no config file has been found.
        """
        if isinstance(vocab_path, str):
            vocab_path = Path(vocab_path)

        if not vocab_path.exists():
            raise ValueError(f"Can not find configuration file at {vocab_path}")

        vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
        return WordGPTTokenizer.from_dict(vocabulary)

    @property
    def vocabulary(self):
        return self._atoi.keys()

    def tokenize(self, text: str) -> torch.Tensor:
        tokens = text.split()
        token_ids = [self._atoi[token] for token in tokens]

        return torch.tensor(token_ids)

    def decode(self, token_ids: torch.Tensor) -> str:
        ids = token_ids.cpu().tolist()
        tokens = [self._itoa[token_id] for token_id in ids]

        return " ".join(tokens)
