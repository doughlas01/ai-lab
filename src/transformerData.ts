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

export type TokenMatch = {
  targetIndex: number;
  keyLabel: string;
  weight: number;
  reason: string;
};

export type TokenInspection = {
  token: string;
  queryQuestion: string;
  keys: TokenMatch[];
  valuePayload: string;
  synthesis: string;
};

export const tokenDetectiveData: TokenInspection[] = [
  {
    token: 'The',
    queryQuestion: '“I am a definite article. Which upcoming noun do I specify?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Article / Determiner', weight: 0.12, reason: 'Self-position reference' },
      { targetIndex: 1, keyLabel: 'Subject / Animate Noun', weight: 0.86, reason: 'Target noun being specified' },
      { targetIndex: 2, keyLabel: 'Main Action Verb', weight: 0.15, reason: 'Clause predicate' },
      { targetIndex: 3, keyLabel: 'Causal Conjunction', weight: 0.05, reason: 'Subordinate clause connector' },
      { targetIndex: 4, keyLabel: 'Pronoun', weight: 0.08, reason: 'Future coreferent' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.04, reason: 'Temporal marker' },
      { targetIndex: 6, keyLabel: 'Predicate Adjective', weight: 0.09, reason: 'Final state' }
    ],
    valuePayload: 'Definite entity: a specific, known feline subject.',
    synthesis: '“The” binds to “cat”, establishing that the narrative refers to a specific animal rather than any generic cat.'
  },
  {
    token: 'cat',
    queryQuestion: '“I am the main subject noun. What action did I take, and what state am I in?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Definite Determiner', weight: 0.28, reason: 'Direct modifier' },
      { targetIndex: 1, keyLabel: 'Subject / Animate Noun', weight: 0.18, reason: 'Self-identity' },
      { targetIndex: 2, keyLabel: 'Main Action Verb', weight: 0.92, reason: 'Direct action performed (sat)' },
      { targetIndex: 3, keyLabel: 'Causal Conjunction', weight: 0.22, reason: 'Reasoning link' },
      { targetIndex: 4, keyLabel: 'Subject Pronoun', weight: 0.35, reason: 'Future coreference target' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.16, reason: 'State verb' },
      { targetIndex: 6, keyLabel: 'Predicate Adjective', weight: 0.78, reason: 'Condition/state (tired)' }
    ],
    valuePayload: 'Entity traits: resting feline, seated position, fatigue condition.',
    synthesis: '“cat” absorbs its action (“sat”) and condition (“tired”) so downstream layers know the full context of the subject.'
  },
  {
    token: 'sat',
    queryQuestion: '“I am the action verb. Who sat down, and why did they sit?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Definite Determiner', weight: 0.14, reason: 'Subject modifier' },
      { targetIndex: 1, keyLabel: 'Subject / Agent', weight: 0.94, reason: 'The agent who sat down' },
      { targetIndex: 2, keyLabel: 'Action Verb', weight: 0.16, reason: 'Self-action' },
      { targetIndex: 3, keyLabel: 'Causal Conjunction', weight: 0.81, reason: 'Points to explanation clause' },
      { targetIndex: 4, keyLabel: 'Subject Pronoun', weight: 0.29, reason: 'Agent reference in cause clause' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.21, reason: 'Tense anchor' },
      { targetIndex: 6, keyLabel: 'Underlying Cause', weight: 0.68, reason: 'The root cause of sitting' }
    ],
    valuePayload: 'Action: sitting posture; Agent: cat; Root Cause: fatigue.',
    synthesis: '“sat” ties the actor (“cat”) to the rationale (“because it was tired”), encoding a complete causal action.'
  },
  {
    token: 'because',
    queryQuestion: '“I am a causal conjunction. Which action am I explaining with what cause?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Article', weight: 0.08, reason: 'Background context' },
      { targetIndex: 1, keyLabel: 'Agent', weight: 0.38, reason: 'Entity involved' },
      { targetIndex: 2, keyLabel: 'Prior Event / Effect', weight: 0.89, reason: 'The event being justified (sat)' },
      { targetIndex: 3, keyLabel: 'Causal Conjunction', weight: 0.14, reason: 'Self-operator' },
      { targetIndex: 4, keyLabel: 'Subject Pronoun', weight: 0.44, reason: 'Subject of cause clause' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.31, reason: 'Tense alignment' },
      { targetIndex: 6, keyLabel: 'Subsequent Cause', weight: 0.93, reason: 'The explanatory condition (tired)' }
    ],
    valuePayload: 'Causal relationship: [cat sitting] caused by [fatigue].',
    synthesis: '“because” bridges the two clauses, pulling the effect from the left and the cause from the right.'
  },
  {
    token: 'it',
    queryQuestion: '“I am a singular neuter pronoun. What antecedent noun do I refer to?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Article', weight: 0.11, reason: 'Determiner of antecedent' },
      { targetIndex: 1, keyLabel: 'Antecedent Noun', weight: 0.95, reason: 'EXACT ANTECEDENT MATCH: cat!' },
      { targetIndex: 2, keyLabel: 'Prior Action', weight: 0.34, reason: 'Action previously performed' },
      { targetIndex: 3, keyLabel: 'Causal Conjunction', weight: 0.24, reason: 'Clause marker' },
      { targetIndex: 4, keyLabel: 'Subject Pronoun', weight: 0.15, reason: 'Self-reference' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.42, reason: 'Immediate predicate' },
      { targetIndex: 6, keyLabel: 'Predicate Adjective', weight: 0.72, reason: 'Condition describing this pronoun' }
    ],
    valuePayload: 'Resolved identity: the cat; current condition: tired.',
    synthesis: '“it” resolves coreference to “cat”. Without attention, “it” is ambiguous; with attention, it assumes the cat’s identity.'
  },
  {
    token: 'was',
    queryQuestion: '“I am a linking copula. Which subject pronoun am I connecting to which state?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Article', weight: 0.06, reason: 'Distant determiner' },
      { targetIndex: 1, keyLabel: 'Root Entity', weight: 0.62, reason: 'Underlying entity' },
      { targetIndex: 2, keyLabel: 'Prior Action', weight: 0.25, reason: 'Previous clause verb' },
      { targetIndex: 3, keyLabel: 'Clause Conjunction', weight: 0.35, reason: 'Local clause head' },
      { targetIndex: 4, keyLabel: 'Local Subject', weight: 0.88, reason: 'Subject being linked (it)' },
      { targetIndex: 5, keyLabel: 'Auxiliary Verb', weight: 0.12, reason: 'Self-reference' },
      { targetIndex: 6, keyLabel: 'Predicate Adjective', weight: 0.91, reason: 'Complement state (tired)' }
    ],
    valuePayload: 'Predication: [it/cat] was in state of [tiredness].',
    synthesis: '“was” binds the local subject (“it”) to its adjective complement (“tired”) in the past tense.'
  },
  {
    token: 'tired',
    queryQuestion: '“I am an adjective denoting exhaustion. Who is tired and what did that cause?”',
    keys: [
      { targetIndex: 0, keyLabel: 'Determiner', weight: 0.08, reason: 'Background determiner' },
      { targetIndex: 1, keyLabel: 'Core Entity', weight: 0.85, reason: 'The real-world entity that is exhausted' },
      { targetIndex: 2, keyLabel: 'Resulting Action', weight: 0.76, reason: 'The action caused by this state (sat)' },
      { targetIndex: 3, keyLabel: 'Causal Reason', weight: 0.58, reason: 'Explains why this state is invoked' },
      { targetIndex: 4, keyLabel: 'Immediate Subject', weight: 0.91, reason: 'Pronoun directly modified (it)' },
      { targetIndex: 5, keyLabel: 'Linking Verb', weight: 0.54, reason: 'Connecting verb (was)' },
      { targetIndex: 6, keyLabel: 'Predicate Adjective', weight: 0.16, reason: 'Self-state' }
    ],
    valuePayload: 'Exhaustion condition; Owner: cat; Consequence: sitting.',
    synthesis: '“tired” grounds its meaning in “cat” and provides the causal explanation for the action “sat”.'
  }
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

