# WordGPT

It's an implementation from scratch of a _decoder only_ transformer model, with a GPT-like architecture and custom word-based tokenizer. The model is trained on the [WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/).

## Useful formulas

$$
\begin{align}
Attention(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \\

PE(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
&= sin(pos \times 10000^{-2i/d_{model}}) \\

PE(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
&= cos(pos \times 10000^{-2i/d_{model}}) \\
\end{align}
$$
