Automatic speech recognition is a deep-learning application that enables the automatic transcription and translation of spoken language into text.
The task is the transcription of the LJ Speech dataset (v 1.1).
This small dataset has around 13000 short audio clips and the corresponding transcription of reading passages from several books.

The challenge proposed to implement the forward pass of different decoder implementations and compare their performance,
two different encoder-decoder architectures to solve this problem:
  1. The decoder employs a recurrent-based approach to process the input
     class TextDecoderRecurrent(torch.nn.Module)
  3. The decoder employs an attention-based mechanism to process the input
     class TextDecoderTransformer(torch.nn.Module)

The encoder’s architecture is based on the Transformer, comprising an embedding mechanism followed by a stack of N_enc = 4 identical blocks.

<img width="543" alt="image" src="https://github.com/tomerkl22/Automatic-Speech-Recognition/assets/94317058/cdf91d72-b24f-4414-aa43-9570fee17a74">



Solution: 

Different metrics for comparing texts are presented:

  Results on test set with RNN:
  
  <img width="234" alt="image" src="https://github.com/tomerkl22/Automatic-Speech-Recognition/assets/94317058/1b74f8f2-77c7-4e6b-a3c7-2559ded9937c">
  
  Results on test set with attention:
  
  <img width="231" alt="image" src="https://github.com/tomerkl22/Automatic-Speech-Recognition/assets/94317058/21ca852e-aff5-4e6d-b59e-67373ee98ae8">


Results have shown us that the decoder with self-attention mechanisms have better performance compared to RNN.
It is hard to have long dependencies in tokens using RNNs. On the other hand, attention-based mechanisms do not suffer from this problem since each token can be accessed in constant time. Therefore, the sequences are processed as a whole (rather than token by token), which avoids long-term dependency issues.
