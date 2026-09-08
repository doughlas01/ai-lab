export type TransformerLesson = {
  id: string;
  kicker: string;
  title: string;
  question: string;
  intro: string;
  sections: { heading: string; body: string; formula?: string; bullets?: string[] }[];
  takeaway: string;
};

export const transformerLessons: TransformerLesson[] = [
  { id: 'tokens', kicker: '01 / Input', title: 'From text to tokens', question: 'How does a Transformer see a sentence?', intro: 'A Transformer does not read words as human-readable strings. It receives a sequence of token IDs, then turns those IDs into vectors that can be processed by neural layers.', sections: [
    { heading: 'Tokens are the interface', body: 'A tokenizer splits text into tokens. A token may be a whole word, part of a word, punctuation, or whitespace. The model processes the resulting sequence of IDs rather than the original characters directly.' },
    { heading: 'Embeddings give tokens coordinates', body: 'An embedding table maps each token ID to a learned vector. Similar usage patterns can lead to useful geometric relationships, but the vector is not a dictionary definition.' },
    { heading: 'Why this matters', body: 'Every later stage operates on these vectors. The sequence length, embedding width, and batch size determine the shape of the data flowing through the model.', bullets: ['Text → tokenizer → token IDs.', 'Token IDs → embedding vectors.', 'Vectors become the input to the Transformer blocks.'] }
  ], takeaway: 'A Transformer starts with token IDs converted into vectors that neural layers can transform.' },
  { id: 'position', kicker: '02 / Order', title: 'How does it know order?', question: 'Why is “dog bites man” different from “man bites dog”?', intro: 'Attention can compare tokens, but the model also needs information about where each token appears in the sequence.', sections: [
    { heading: 'Attention alone is not enough', body: 'If the same token vectors were presented with no position information, the model would have less information about sequence order. Position information gives each location a distinguishable signal.' },
    { heading: 'Position enters the representation', body: 'Architectures can add or combine positional information in different ways. The important intuition is that token content and token location travel together through the block.' },
    { heading: 'The practical consequence', body: 'Changing the order of tokens changes the vectors seen by later computation. This lets the model represent syntax, relationships, and sequence-dependent meaning.', bullets: ['Content answers “what token is this?”', 'Position answers “where is it?”', 'Both influence the representations passed forward.'] }
  ], takeaway: 'Transformers need a way to represent sequence order alongside token content.' },
  { id: 'attention', kicker: '03 / Relate', title: 'Self-attention connects tokens', question: 'How can one token use information from the rest of the sequence?', intro: 'Self-attention lets each position form a weighted interaction with other positions. The current position creates a Query, while the sequence supplies Keys and Values to compare and retrieve.', sections: [
    { heading: 'Queries ask', body: 'The Query represents what the current position is looking for. A compatibility calculation compares that Query with Keys from other positions.' },
    { heading: 'Keys match and Values carry', body: 'Keys help determine which positions are relevant. Values carry the information that is mixed into the output. Different heads can learn different relationship patterns.' },
    { heading: 'The result is contextual', body: 'A token representation after attention is no longer based only on that token. It includes selected information from the surrounding sequence.', formula: 'Attention(Q, K, V) = softmax(QKᵀ / √d) V', bullets: ['Q: what this position needs.', 'K: what each position offers for matching.', 'V: information retrieved after matching.'] }
  ], takeaway: 'Self-attention builds contextual representations by matching Queries with Keys and mixing the corresponding Values.' },
  { id: 'block', kicker: '04 / Transform', title: 'Inside one Transformer block', question: 'What happens after attention?', intro: 'A modern Transformer block typically combines an attention sublayer with a position-wise feed-forward network, wrapped with residual connections and normalization.', sections: [
    { heading: 'Attention mixes information', body: 'The attention sublayer lets positions exchange information. It changes each position using information from other positions in the sequence.' },
    { heading: 'The MLP transforms features', body: 'The feed-forward or MLP sublayer operates on each position independently. It expands and transforms the feature representation, then projects it back to the model width.' },
    { heading: 'Residuals protect the signal', body: 'Residual connections add a block’s update to the representation that entered it. This gives information a shorter path through many layers and helps optimization.', bullets: ['Normalize the representation.', 'Mix across positions with attention.', 'Transform features with the MLP.', 'Add updates through residual paths.'] }
  ], takeaway: 'A Transformer block alternates communication across positions with feature transformation at each position.' },
  { id: 'heads', kicker: '05 / Parallel', title: 'Why multiple attention heads?', question: 'Why not use one attention pattern?', intro: 'Multi-head attention gives the model several learned subspaces in which to compare and retrieve information at the same time.', sections: [
    { heading: 'Different views of a sequence', body: 'One head may learn local word relationships, another may track a name or reference, and another may respond to a longer structural pattern. These are illustrative intuitions, not fixed assignments.' },
    { heading: 'Split, attend, combine', body: 'The model projects the representation into multiple head-specific Query, Key, and Value spaces. Each head performs attention, and the results are combined and projected back.' },
    { heading: 'More heads are not automatically better', body: 'Heads add representational pathways and computation. Their usefulness depends on model width, training, task, and implementation.', bullets: ['Heads operate in parallel.', 'Each head can learn a different interaction pattern.', 'The outputs are combined into the block representation.'] }
  ], takeaway: 'Multiple heads let attention examine relationships through several learned representation spaces.' },
  { id: 'stack', kicker: '06 / Compose', title: 'Why stack many blocks?', question: 'How does a shallow operation become a capable model?', intro: 'One block makes a local update to the sequence representation. Stacking blocks lets later layers operate on increasingly transformed and contextual features.', sections: [
    { heading: 'Representations become richer', body: 'Early layers may capture relatively local or simple patterns. Deeper layers can combine earlier signals into more abstract relationships. Exact layer roles are learned rather than manually assigned.' },
    { heading: 'Information flows through depth', body: 'Residual pathways carry a running representation while each block contributes an update. The model repeatedly refines the sequence state.' },
    { heading: 'The final head produces a result', body: 'For language modeling, the final representation at a position is mapped to scores over the vocabulary. During generation, the model selects or samples the next token.', bullets: ['Input representations enter block 1.', 'Each block adds an attention and MLP update.', 'The final representation supports prediction.'] }
  ], takeaway: 'Depth lets a Transformer repeatedly refine contextual representations before making a prediction.' }
];

export const researchPapers = [
  {
    year: '2017',
    authors: 'Vaswani et al. · Google Brain / Google Research',
    title: 'Attention Is All You Need',
    citation: 'NeurIPS 2017',
    question: 'Can sequence transduction work without recurrence or convolution?',
    summary: 'This paper introduced the Transformer: an encoder-decoder architecture built around attention, removing recurrent and convolutional sequence-processing components from the main architecture.',
    ideas: [
      ['Self-attention', 'Each position can connect to other positions in the sequence and gather contextual information.'],
      ['Multi-head attention', 'Several learned attention subspaces process different relationship patterns in parallel.'],
      ['Position information', 'Since the architecture does not process tokens sequentially by recurrence, positional encodings provide order information.'],
      ['Parallel training', 'Attention allows sequence positions to be processed more in parallel during training than recurrent approaches.']
    ],
    caveat: 'This paper introduced the original Transformer for sequence transduction. It did not introduce today’s decoder-only LLM recipe, KV-cache serving systems, or every later attention variant.'
  }
];
