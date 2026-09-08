export type Reading = {
  intro: string;
  sections: { heading: string; body: string; bullets?: string[]; formula?: string }[];
};

export const reading: Record<string, Reading> = {
  'kv-cache': {
    intro: 'A KV cache is the reusable attention history carried forward during autoregressive generation.',
    sections: [
      { heading: 'Start with Q, K, and V', body: 'When a Transformer performs attention, each token is transformed into a Query, Key, and Value. A Query asks what information the current step needs. Keys help identify what each earlier token represents. Values contain the information that can be retrieved.' },
      { heading: 'Why cache it?', body: 'The model generates one token at a time. Without caching, it could recompute the Key and Value representations for every previous token at every step. Instead, those representations remain ready in memory for future decoding steps.' },
      { heading: 'A concrete conversation', body: 'Imagine the user says: “My name is Alex.” The token positions produce My → K₁ V₁, name → K₂ V₂, is → K₃ V₃, Alex → K₄ V₄. As the assistant continues, each new token adds another pair to the stored history.', bullets: ['The cache stores activations, not the whole model.', 'Caching saves repeated computation but consumes extra memory.', 'The exact layers and heads that cache information depend on the architecture.'] }
    ]
  },
  growth: {
    intro: 'The token-dependent portion of KV memory grows approximately linearly with the number of cached positions.',
    sections: [
      { heading: 'The first important relationship', body: 'Every additional cached token adds a Key and a Value at every layer and KV head. Double the context under the same model assumptions and the token-dependent cache approximately doubles.', formula: 'KV bytes ≈ 2 × L × H_KV × D_head × T × B' },
      { heading: 'Read the symbols', body: 'L is the number of layers, H_KV is the number of Key/Value heads, D_head is the dimension per head, T is the number of cached tokens, and B is bytes per element. The factor of two represents one Key tensor and one Value tensor.' },
      { heading: 'Why serving feels the pressure', body: 'A server holds model weights plus separate active-request state. Ten short conversations may fit comfortably, while a smaller number of long research sessions can consume much more KV memory. In the simulator, 1,280 token positions are about 128 times the 10-token example, but this is not a claim that total GPU memory is exactly 128 times larger.', bullets: ['More context means more historical information to represent.', 'More users multiply the per-user cache.', 'Weights, runtime buffers, fragmentation, and scheduling also consume memory.'] }
    ]
  },
  window: {
    intro: 'Sliding-window attention deliberately limits direct attention to a bounded region of recent tokens.',
    sections: [
      { heading: 'Watch the window move', body: 'With a window size of four, a sequence might attend to tokens 7, 8, 9, and 10. When token 11 arrives, the active region becomes 8, 9, 10, and 11. The window slides forward as generation continues.' },
      { heading: 'What it saves', body: 'The active token history is bounded by the window size rather than the full conversation length for that local-attention mechanism. A smaller window can reduce the directly retained token-level memory.' },
      { heading: 'What it gives up', body: 'A token outside the window is not directly accessible to that sliding-window attention layer. If an access code appeared thousands of tokens ago, the local layer cannot simply look it up from its old position. Other layers or mechanisms may still preserve long-term information.', bullets: ['Larger window: more memory and more direct history.', 'Smaller window: less memory and less direct distant retrieval.', 'A sliding-window layer does not mean the complete model has no long-term memory.'] }
    ]
  },
  linear: {
    intro: 'Linear-attention methods change how sequence history is accumulated so some formulations can use more favorable sequence-length scaling.',
    sections: [
      { heading: 'The intuition', body: 'Full attention can be pictured as a current Query looking across many token-level records. Linear-attention intuition instead accumulates historical information into a running representation that can be updated and reused.' },
      { heading: 'Why “linear”?', body: 'The name refers to sequence-length scaling in particular formulations. Standard softmax attention has a token-to-token interaction structure with quadratic sequence-length computation. Linear-attention families use different algebraic arrangements or feature mappings to obtain more favorable scaling in the relevant computation.' },
      { heading: 'The retrieval trade-off', body: 'A compact state has a harder job if it must preserve a huge amount of exact detail. This does not mean linear attention always retrieves poorly. Exact behavior depends on the formulation, training procedure, model, and task.', bullets: ['It is a family of methods, not one single algorithm.', 'It is not simply a smaller copy of the standard KV cache.', 'The simulator shows the representation trade-off conceptually, not as a benchmark.'] }
    ]
  },
  ssm: {
    intro: 'State-space models carry sequence history through an evolving hidden state.',
    sections: [
      { heading: 'A state that changes over time', body: 'An SSM reads an input, updates its current state, and produces an output from that state. A conceptual recurrence is: state(t) = transition(state(t − 1), input(t)).' },
      { heading: 'A useful human analogy', body: 'When you read a paragraph, you usually keep an internal representation of who is involved, what happened, and what matters. You do not retain every character separately. That evolving representation is an intuition for a state.' },
      { heading: 'A different mechanism', body: 'SSMs should not be described as compressed attention. They are a different sequence-modeling mechanism with a recurrence structure. A compact state can be efficient, but it has finite capacity and must preserve useful detail through repeated updates.', bullets: ['Token 1 updates State 1.', 'Token 2 updates State 2 using the previous state.', 'The current state encodes some influence from prior inputs.'] }
    ]
  },
  mamba: {
    intro: 'Mamba-style selective state-space models make state processing input-dependent.',
    sections: [
      { heading: 'Not every token deserves equal treatment', body: 'A useful sequence model may process a key fact differently from a formatting detail. In the educational visualization, an important fact can create a stronger conceptual state update while a less relevant token creates a smaller update.' },
      { heading: 'What selectivity means', body: 'The state dynamics and input processing can depend on the current input. This creates a mechanism for selectively preserving or processing information instead of applying exactly the same update behavior to every token.' },
      { heading: 'A careful interpretation', body: 'The visual “influence” value is not a human-readable internal importance score. It represents the intuition of input-dependent state updates. Mamba is based on selective state-space modeling, not standard softmax attention.', bullets: ['Important information → stronger conceptual state update.', 'Less useful information → weaker conceptual state update.', 'The actual behavior depends on the trained model and implementation.'] }
    ]
  },
  tradeoffs: {
    intro: 'Different mechanisms move the system toward different balances. None is a universal winner.',
    sections: [
      { heading: 'Compare the dimensions', body: 'Think about memory pressure, direct retrieval, long-context behavior, latency, throughput, and quality together. A conceptual score in this lab is a teaching aid, not a measured benchmark.' },
      { heading: 'No free lunch', body: 'A full token archive is useful for precise lookup but expensive to carry. A compact notebook is cheaper to carry but may not preserve every exact detail. Architecture choices move the balance between those properties.' },
      { heading: 'What changes the outcome?', body: 'Architecture is only one factor. Training data, parameter count, optimization, positional representation, quantization, inference implementation, hardware, and task distribution all matter.', bullets: ['Do not say more attention always means better quality.', 'Do not say more Mamba always means worse quality.', 'Do not present conceptual meters as public model benchmarks.'] }
    ]
  },
  hybrid: {
    intro: 'Hybrid architectures combine different sequence-processing mechanisms instead of choosing only one everywhere.',
    sections: [
      { heading: 'Two paths, different jobs', body: 'Attention can provide strong token-level interactions. State-based layers can provide compact long-sequence processing. A hybrid can place these mechanisms at different depths and allow each to contribute where it is useful.' },
      { heading: 'A design choice, not a recipe', body: 'The number, placement, and proportion of attention and state-based layers are architectural choices. They can influence memory use, retrieval, latency, throughput, and long-context behavior, but architecture alone does not determine the result.' },
      { heading: 'The reality check', body: 'Public architecture facts and private frontier-model speculation must be kept separate. There is no single Attention:Mamba ratio that can be assumed to be an industry standard.', bullets: ['Attention path → explicit token-level retrieval.', 'State path → compact evolving representation.', 'Hybrid → combine mechanisms and examine the trade-offs.'] }
    ]
  }
};
