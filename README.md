Automatic speech recognition is a deep-learning application that enables the automatic transcription and translation of spoken language into text.
The task is the transcription of the LJ Speech dataset (v 1.1).
This small dataset has around 13000 short audio clips and the corresponding transcription of reading passages from several books.

The challenge proposed to implement the forward pass of different decoder implementations and compare their performance,
two different encoder-decoder architectures to solve this problem:
  1. The decoder employs a recurrent-based approach to process the input
  2. The decoder employs an attention-based mechanism to process the input

The encoder’s architecture is based on the Transformer, comprising an embedding mechanism followed by a stack of N_enc = 4 identical blocks.
