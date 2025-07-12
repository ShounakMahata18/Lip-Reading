class LabelAligner:
    def __init__(self, vocab=None, max_length=40, blank_token="<BLANK>"):
        # Default: a-z, space, blank
        if vocab is None:
            self.vocab = ['<BLANK>'] + [chr(i) for i in range(97, 123)] + [' ']
        else:
            self.vocab = vocab

        self.max_length = max_length
        self.blank_token = blank_token
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text):
        """Convert text to padded/truncated integer vector of length `max_length`."""
        indices = [self.char_to_idx.get(c, 0) for c in text.lower()]
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [self.char_to_idx[self.blank_token]] * (self.max_length - len(indices))
        return indices

    def decode(self, indices):
        """Convert a list of indices back to string (ignoring blanks)."""
        return ''.join([self.idx_to_char[i] for i in indices if self.idx_to_char[i] != self.blank_token])

    def one_hot(self, indices):
        import tensorflow as tf
        return tf.one_hot(indices, depth=len(self.vocab), dtype=tf.float32)
