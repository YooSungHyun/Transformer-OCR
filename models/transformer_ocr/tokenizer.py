from itertools import groupby

import numpy as np

# example = "aAbBcCdDeE.....!@#$%!@...."
# we don't think number. use only string like.
CHARS = ""


class TransformerTokenizer:
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, max_text_length=32):
        assert len(CHARS) == 0, "PLZ input vocab text!!!!"
        self.PAD_TK, self.UNK_TK, self.SOS, self.EOS = "¶", "¤", "@", "#"
        self.chars = [self.PAD_TK] + [self.UNK_TK] + [self.SOS] + [self.EOS] + list(CHARS)
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        text = ["@"] + list(text) + ["#"]
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def batch_decode(self, indices):
        return list(map(lambda x: self.decode(x), indices))

    def decode(self, indices):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in indices if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
