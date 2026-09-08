export type Lesson = {
  id: string;
  kicker: string;
  title: string;
  question: string;
  explanation: string;
  analogy: string;
  visual: string;
  tradeoff: string;
  takeaway: string;
  check: { prompt: string; answer: string };
};

export const lessons: Lesson[] = [
  {
    id: 'kv-cache', kicker: '01 / Remember', title: 'What is the KV cache?', question: 'How can a model reuse the past without recomputing it?',
    explanation: 'Each token becomes a Query, Key, and Value. During generation, previously computed Keys and Values can stay ready for future attention steps. That stored history is the KV cache.',
    analogy: 'Think of a librarian keeping labeled information cards on a shelf. A new token asks a question; Keys help find the right cards and Values contain what can be retrieved.',
    visual: 'Tokens become K / V pairs and collect in a persistent cache.', tradeoff: 'Caching saves repeated computation, but the saved activations require GPU memory.',
    takeaway: 'KV cache stores previously computed attention Key/Value information so the model does not recompute it for every generated token.',
    check: { prompt: 'What does the KV cache store?', answer: 'Cached Key and Value activations for token positions in attention layers.' }
  },
  {
    id: 'growth', kicker: '02 / Scale', title: 'Why does memory grow?', question: 'What changes when the conversation gets longer?',
    explanation: 'Every additional cached token adds another Key and Value position. Under otherwise identical assumptions, the token-dependent portion of KV memory grows approximately linearly with context length.',
    analogy: 'One conversation is one growing shelf. More users mean more shelves, each carrying its own active sequence history.',
    visual: 'The cache blocks grow as context and active users increase.', tradeoff: 'Longer context can provide more usable history, but it reduces how many requests fit on the same GPU.',
    takeaway: 'As more tokens are cached, KV-cache memory grows and puts pressure on GPU memory and serving capacity.',
    check: { prompt: 'Why does a 1,280-token user have roughly 128 times the token positions of a 10-token user?', answer: 'Because 1,280 divided by 10 is 128; this is a proportional token-position comparison, not a claim about total GPU memory.' }
  },
  {
    id: 'window', kicker: '03 / Restrict', title: 'Sliding-window attention', question: 'Can we stop carrying every old token?',
    explanation: 'Sliding-window attention limits direct attention to a fixed-size region of recent tokens. As new tokens arrive, the window moves forward and older positions leave the active region.',
    analogy: 'Instead of keeping the entire conversation open on the desk, keep only the most recent pages directly in front of you.',
    visual: 'A highlighted window slides over a long token stream while old positions fade.', tradeoff: 'A smaller window bounds active memory but makes distant token-level retrieval less direct.',
    takeaway: 'Sliding-window attention controls memory growth by limiting attention to a bounded region of recent tokens.',
    check: { prompt: 'What happens to a token outside the active window?', answer: 'That sliding-window attention layer cannot directly attend to it, although other mechanisms may still preserve information.' }
  },
  {
    id: 'linear', kicker: '04 / Compress', title: 'Linear attention', question: 'Can history become a running representation?',
    explanation: 'Linear-attention methods reformulate attention so historical information can be accumulated into a more compact representation in relevant formulations. “Linear attention” is a family of methods, not one algorithm.',
    analogy: 'Instead of keeping every page open, maintain a running set of notes that can be updated as each new page arrives.',
    visual: 'Many token records converge into a compact state object.', tradeoff: 'A compact representation can use less memory, but exact token-level lookups may be harder depending on the formulation and task.',
    takeaway: 'Linear-attention methods change how history is processed so it can be accumulated more compactly or with more favorable sequence-length scaling.',
    check: { prompt: 'Why is linear attention not simply a smaller KV cache?', answer: 'It generally changes the computation or representation of sequence history instead of storing the same full attention cache in compressed form.' }
  },
  {
    id: 'ssm', kicker: '05 / Evolve', title: 'State-space models', question: 'What if history travels as an evolving state?',
    explanation: 'State-space models process sequences by maintaining and updating a hidden state. The state carries information forward rather than exposing every old token as a separate attention lookup record.',
    analogy: 'When you finish a paragraph, you usually retain a useful internal summary rather than every character. That evolving summary is an intuition for state.',
    visual: 'Token 1 updates State 1, then Token 2 updates State 2, and so on.', tradeoff: 'A compact state can be efficient, but it has finite capacity and must preserve useful detail through its updates.',
    takeaway: 'SSMs carry sequence history through an evolving hidden state rather than relying entirely on explicit token-to-token attention.',
    check: { prompt: 'What is the core object an SSM updates over time?', answer: 'A hidden state that carries information from earlier inputs.' }
  },
  {
    id: 'mamba', kicker: '06 / Select', title: 'Mamba-style selectivity', question: 'Should every token update memory equally?',
    explanation: 'Mamba-style selective state-space models make state processing depend on the current input. The teaching intuition is that useful information can influence the state more strongly than less relevant information.',
    analogy: 'A reader underlines a key number in a report but gives less attention to formatting details.',
    visual: 'Tokens enter a state-update region with different conceptual influence strengths.', tradeoff: 'Selective state updates aim to spend representation capacity where it matters, but the visual influence values are educational abstractions, not model internals.',
    takeaway: 'Mamba introduces selective, input-dependent state dynamics so information can be processed and retained differently depending on the input.',
    check: { prompt: 'Does the visualization’s “importance” value represent a literal Mamba score?', answer: 'No. It represents the intuition of input-dependent state updates, not a human-readable internal score.' }
  },
  {
    id: 'tradeoffs', kicker: '07 / Balance', title: 'There is no free lunch', question: 'Which property should an architecture prioritize?',
    explanation: 'Architectural choices move a system toward different balances between memory efficiency, precise retrieval, latency, throughput, context handling, and quality. The exact result depends on implementation, training, hardware, and workload.',
    analogy: 'A compact notebook is easy to carry, while a full archive is easier to search precisely. Each is useful in a different situation.',
    visual: 'The comparison meters shift as the selected history representation changes.', tradeoff: 'Do not treat conceptual scores as benchmarks or rank every architecture universally.',
    takeaway: 'Long-context modeling is a trade-off between how much history we preserve, how precisely we retrieve it, and how efficiently we represent it.',
    check: { prompt: 'Does one architecture automatically produce a better model?', answer: 'No. Architecture is one factor among training data, scale, optimization, implementation, hardware, and task distribution.' }
  },
  {
    id: 'hybrid', kicker: '08 / Combine', title: 'Why build hybrids?', question: 'What if different layers use different mechanisms?',
    explanation: 'A hybrid can place attention and state-based layers together. Attention can provide strong token-level interactions, while state-based layers can support compact long-sequence processing.',
    analogy: 'A workshop uses both a microscope for precise inspection and a conveyor for moving large volumes efficiently.',
    visual: 'Input tokens split into explicit-attention and compact-state paths before recombining.', tradeoff: 'The number, placement, and proportion of layer types are design choices, not a universal industry recipe.',
    takeaway: 'Hybrid models combine different sequence-processing mechanisms to balance their strengths and weaknesses.',
    check: { prompt: 'Why might a hybrid be useful?', answer: 'It can combine explicit retrieval capacity with more compact state-based processing, depending on the workload.' }
  }
];
