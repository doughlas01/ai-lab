# Interactive AI Systems Lab — KV Cache & Token Compression

## 1. Document Purpose

This document is a build specification for an interactive, browser-based educational website focused on one topic only:

> **How long-context LLM inference creates KV-cache pressure, and how different architectural approaches attempt to reduce or bound the cost of representing historical tokens.**

The first release must be a polished, interactive learning experience—not a general AI education platform. The architecture should be extensible so additional topics can be added later without rewriting the core system.

The experience should feel closer to an interactive technical laboratory than a slide deck. Users should be able to manipulate variables and immediately see the consequences in 3D graphics, charts, memory meters, and animations.

Primary audience:
- Junior ML / GenAI engineers
- Software engineers learning LLM inference
- Students with basic Transformer/LLM familiarity
- Engineers who know the terms "attention" and "KV cache" but do not yet have a systems-level mental model

Primary learning objective:
- A learner should finish the module understanding **why KV cache grows with context**, **why this matters for serving**, and **how sliding-window attention, linear attention, SSMs, and Mamba-style selective state approaches represent history differently**.

---

# 2. Product Philosophy

## 2.1 Core principle

Do not teach this topic primarily through paragraphs.

Teach it through:

1. **Visualization** — make invisible memory and information flow visible.
2. **Interaction** — let the learner change context length, users, window size, and architecture.
3. **Immediate feedback** — every change should update the 3D scene and quantitative indicators in real time.
4. **Progressive disclosure** — start intuitive, then reveal equations/details when the learner requests them.
5. **Experimentation** — the learner should be able to intentionally create both good and bad configurations.

## 2.2 Mental model the website should build

The learner should leave with this chain in their head:

```text
More context
    ↓
More historical information to represent
    ↓
Standard attention keeps K/V representations for prior tokens
    ↓
KV cache grows with context length
    ↓
Long contexts reduce concurrent serving capacity
    ↓
Researchers explore alternative ways to represent history
    ↓
Sliding Window / Linear Attention / SSM / Mamba-style approaches
    ↓
Different trade-offs between memory, retrieval, latency, and context handling
    ↓
Hybrid designs can combine mechanisms
```

## 2.3 Educational rule

Never imply that one architecture universally dominates the others.

The UI must communicate **trade-offs**, not a simplistic ranking.

Avoid claims such as:
- "More Mamba always means worse quality."
- "More attention always means better quality."
- "All frontier models use the same hybrid ratio."

The website should distinguish:
- conceptual behavior
- simulated metrics
- real measured benchmarks
- public architecture facts

For the first version, the metrics are primarily **educational simulations derived from transparent formulas**, not claims about proprietary model quality.

---

# 3. Scope of Version 1

## 3.1 In scope

The website should teach:

1. What KV cache is at a conceptual level.
2. Why KV cache grows with token count.
3. Why this affects GPU memory and concurrent serving.
4. Why compressing across tokens is a distinct problem from compressing across heads.
5. Sliding-window attention.
6. Linear attention as a conceptual state-compression mechanism.
7. State-space models (SSMs) as evolving state representations.
8. Mamba-style selective/input-dependent state updates.
9. Why different mechanisms involve trade-offs.
10. Why hybrid architectures are attractive.
11. A final interactive architecture playground.

## 3.2 Explicitly out of scope for Version 1

Do not build full lessons for:
- FlashAttention
- PagedAttention
- GQA / MQA / MLA in detail
- Continuous batching
- Speculative decoding
- Quantization
- MoE
- Distributed inference
- GPU kernel internals
- Training optimization
- RAG internals
- Model leaderboards

The site architecture should make these possible later, but the first release must remain focused.

---

# 4. Overall Experience

The website should be a single guided interactive module with a strong visual narrative.

Recommended route structure:

```text
/
  → Landing / Module Overview

/kv-cache
  → Main learning experience

/kv-cache/playground
  → Interactive architecture playground

/kv-cache/reference
  → Equations + glossary + assumptions
```

A single-page experience can also be used initially. Prefer a route-based architecture only if it improves code organization.

---

# 5. Visual Direction

## 5.1 Desired aesthetic

The visual language should feel like:
- premium technical visualization
- modern engineering laboratory
- dark, immersive interface
- subtle grid / depth / glow
- clean typography
- restrained use of color
- motion used to explain causality

Avoid:
- childish cartoon style
- excessive neon/cyberpunk decoration
- giant blocks of text
- random floating 3D objects
- visual effects that do not communicate a concept

## 5.2 Layout philosophy

Use a hybrid 2D + 3D interface.

The 3D canvas is the main explanatory area.

2D overlays provide:
- controls
- labels
- equations
- metric cards
- explanations
- progress/navigation

Conceptual layout:

```text
┌───────────────────────────────────────────────────────────────┐
│ AI SYSTEMS LAB                         Module 2 / KV CACHE    │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│                    3D VISUALIZATION                           │
│                                                               │
│          tokens  →  attention  →  cache / state              │
│                                                               │
│                                                               │
│                                                               │
├───────────────────────────────┬───────────────────────────────┤
│ CONTROLS                      │ LIVE METRICS                  │
│ Context Length [slider]       │ KV Memory     3.2 GB         │
│ Users [slider]                │ Users Served  34             │
│ Window Size [slider]          │ Context       32K            │
│ Architecture [dropdown]       │ Retrieval     simulated      │
└───────────────────────────────┴───────────────────────────────┘
```

## 5.3 Color semantics

Choose a consistent semantic palette and keep it stable across the entire module.

Suggested semantic mapping:
- Query (Q): one distinct accent
- Key (K): second accent
- Value (V): third accent
- KV Cache: a persistent storage color
- Active token: bright/high-emphasis treatment
- Forgotten/evicted token: dim / desaturated
- Compressed state: compact glowing structure
- Selective/important information: highlighted state
- Warning / memory pressure: warm warning treatment

Do not rely on color alone. Use labels, shapes, glow intensity, motion, and icons as redundant cues.

---

# 6. Recommended Technology Stack

## Frontend

Preferred:
- React
- TypeScript
- Vite or Next.js
- Three.js
- React Three Fiber for 3D scene composition
- Drei for useful R3F helpers
- Framer Motion or Motion for 2D UI transitions
- Zustand for lightweight global simulation state

Alternative state management is acceptable, but keep the state model centralized.

## Styling

Use:
- CSS Modules, Tailwind, or another maintainable design system
- CSS variables for theme tokens
- responsive layouts

Avoid introducing a heavy UI framework unless it materially accelerates development.

## 3D rendering

Three.js / React Three Fiber.

Use WebGL as the default. WebGPU should be considered later, not required for V1.

## Charts

Use a lightweight charting library only where helpful. Simple canvas/SVG charts can be written directly.

## Deployment

The initial site must be statically deployable.

Possible targets:
- Vercel
- Netlify
- Cloudflare Pages
- GitHub Pages for a static build

No backend is required for Version 1.

---

# 7. Core Architecture

Design the implementation so the educational simulation engine is independent of the 3D renderer.

Recommended architecture:

```text
UI / Controls
     ↓
Learning Module State
     ↓
Simulation Engine
     ├── KV Cache Model
     ├── Sliding Window Model
     ├── Linear Attention Model
     ├── SSM Model
     ├── Selective State Model
     └── Metric Calculator
     ↓
Presentation State
     ├── 3D scene
     ├── 2D metrics
     ├── charts
     └── explanatory annotations
```

The critical rule is:

> **The simulation model must not depend on Three.js.**

This means later the same simulation can power:
- 2D views
- 3D views
- benchmark mode
- quizzes
- future modules

---

# 8. Simulation State Model

Create a typed central model similar to:

```ts
interface SimulationState {
  tokenCount: number;
  activeUsers: number;
  architecture: ArchitectureMode;
  windowSize: number;

  model: {
    layers: number;
    kvHeads: number;
    headDim: number;
    bytesPerElement: number;
  };

  kv: {
    totalElements: number;
    bytes: number;
    gb: number;
  };

  serving: {
    gpuMemoryGb: number;
    modelMemoryGb: number;
    availableKvMemoryGb: number;
    estimatedUsers: number;
  };

  educational: {
    retrievalScore: number;
    memoryScore: number;
    latencyScore: number;
  };
}

type ArchitectureMode =
  | 'standard-attention'
  | 'sliding-window'
  | 'linear-attention'
  | 'ssm'
  | 'mamba-style'
  | 'hybrid';
```

The exact interfaces can change, but the separation should remain.

---

# 9. KV Cache Calculation Engine

The simulator should support a transparent educational estimate.

For a standard decoder-style attention layer, a simplified KV-cache byte estimate can be expressed as:

```text
KV bytes per token
≈ 2 × number_of_KV_heads × head_dim × bytes_per_element × number_of_layers
```

Then:

```text
Total KV bytes
≈ context_tokens × KV bytes per token
```

For multiple simultaneous users:

```text
Total active KV bytes
≈ total KV bytes per user × active users
```

The UI should expose these assumptions.

Example default educational model configuration:

```text
Layers:            32
KV heads:          8
Head dimension:    128
Element size:      2 bytes (FP16/BF16-like estimate)
GPU memory:        24 GB
Model memory:      10 GB (educational placeholder)
```

These are simulation defaults, not claims about a specific commercial model.

Use explicit math in the reference panel and allow the learner to expand "How is this number calculated?".

---

# 10. Example Scenario

Use one consistent example throughout the module so learners build intuition.

### Primary scenario

Two users share one GPU.

User A:
- context = 10 tokens

User B:
- context = 1,280 tokens

The simulator should display:

```text
User A
10 tokens
↓
Small cache

User B
1,280 tokens
↓
~128× token positions vs User A
```

Important wording:

> "All else equal, the token-dependent portion of KV-cache storage scales roughly linearly with context length."

Do not claim that total GPU memory usage is exactly 128× because model weights, runtime buffers, fragmentation, batching effects, etc. may not scale the same way.

---

# 11. Learning Flow

The entire experience should follow this narrative.

## Step 1 — Hook

Question displayed:

> **Why can a GPU serve many short conversations but struggle when everyone starts using long contexts?**

Animation:
- 5–10 short conversations flowing into a GPU
- then gradually replace them with long conversations
- cache blocks grow
- memory meter approaches capacity

CTA:

> **See where the memory goes →**

---

# 12. Scene A — What Is KV Cache?

## Goal

Give a conceptual explanation without overloading the learner with equations.

## 3D objects

Represent a short token sequence as 3D capsules or cards:

```text
[The] [cat] [sat] [on] [the] [mat]
```

As the model processes tokens, create K and V objects behind each token.

Visually:

```text
Token
  │
  ├── K
  └── V
```

The K/V objects move into a cache container.

## Animation

1. Token appears.
2. K and V are generated.
3. They travel to cache.
4. Next token appears.
5. Process repeats.
6. User presses "Generate next token".
7. Existing K/V stays in cache while the new token is added.

## Interaction

- Play / pause
- Step forward one token
- Reset
- Hover token to show K/V
- Toggle "show K" and "show V"

## Teaching text

Keep text short:

> "During autoregressive generation, previously computed Key and Value representations can be reused rather than recomputed from scratch. The KV cache stores them for subsequent decoding steps."

---

# 13. Scene B — Context Growth

## Goal

Make linear cache growth visually undeniable.

Controls:

```text
Context Length
[10] [100] [1K] [4K] [16K] [32K]
```

When the slider moves:
- token stream grows
- cache fills
- memory meter changes
- graph updates in real time

## 3D visualization

Use a long horizontal/curved timeline of token nodes.

Cache blocks should physically accumulate.

At high token counts, do not create tens of thousands of individual Three.js meshes. Use instancing / batching or a symbolic representation.

Example:
- 10–200 tokens: individual token objects
- >200 tokens: grouped token clusters with aggregate labels
- >4K: compressed visual bands with selected representative tokens

This is important for performance.

---

# 14. Scene C — Multi-User GPU

## Goal

Teach that inference serving is a memory-allocation problem across users.

Create a 3D GPU container.

Inside it:

```text
GPU MEMORY
│
├── Model weights
├── User A KV
├── User B KV
├── User C KV
└── User D KV
```

Each user gets a colored lane/region.

Controls:

```text
Number of users:     [1 ─────●──── 100]
Average context:     [1K ─────●─── 128K]
GPU memory:          [8 ──●─────── 80 GB]
```

The memory bar must update immediately.

## Event behavior

When capacity is exceeded:
- subtle warning animation
- memory meter turns critical
- user lanes begin showing "cannot fit" or "requires scheduling" state

Do not simulate production scheduling complexity in V1.

The objective is simply:

> Long context per user can reduce the number of users that fit concurrently.

---

# 15. Scene D — The Central Question

Full-screen transition:

> **Do we really need to represent every old token in exactly the same way?**

Then reveal four routes:

```text
FORGET      COMPRESS        EVOLVE        SELECT
Window      Linear          SSM           Mamba-style
```

This becomes the navigation hub for the architecture techniques.

---

# 16. Scene E — Sliding-Window Attention

## Visual

Create a long token strip.

Use a translucent attention window.

Example:

```text
1 2 3 4 5 6 7 8 9 10
        └── window ──┘
```

Window size slider:

```text
Window size: [4 ───────────── 128]
```

The window moves as tokens are generated.

Old tokens outside the window become dim and/or leave the cache container.

## Interaction

User can:
- increase/decrease window size
- generate tokens
- click an old token
- ask "Can attention access this?"

If the token is outside the window:

```text
ACCESS: NO
Reason: outside active attention window
```

If inside:

```text
ACCESS: YES
```

## Educational challenge

Prompt:

> "The model learned 'My dog's name is Bruno' 5,000 tokens ago. Set the window so that the information is still directly accessible."

The learner adjusts the window.

---

# 17. Scene F — Linear Attention

## Visual goal

Show the transition from many token-level records to a compact state.

Animation:

```text
Token stream
● ● ● ● ● ● ● ● ● ● ●
          ↓
          ↓
      ┌───────────┐
      │  STATE    │
      └───────────┘
```

Old tokens should visually converge into one evolving state object.

## Interaction

A "compression" control can control how aggressively the visualization summarizes the sequence.

Show conceptual metrics:

```text
State size:          constant / bounded in simulation
Token records:       10,000
Direct token access: reduced
```

## Important technical note

Do not implement this as a production linear-attention model unless explicitly added later.

For V1, the simulator is an explanatory abstraction.

The reference panel should explain that linear attention modifies the attention computation so historical information can be accumulated into a running representation, avoiding explicit token-by-token storage in the same form as standard attention.

---

# 18. Scene G — State-Space Models

## Visual

Show a state node evolving through time.

```text
Token 1 → State 1
             ↓
Token 2 → State 2
             ↓
Token 3 → State 3
             ↓
...
```

The state should visibly change shape or internal particles as tokens arrive.

## Interaction

Click any token to see:

```text
This token influenced the state.
```

Click the state to show:

```text
Current state summarizes information from prior sequence history.
```

## Teaching objective

The learner should understand the conceptual difference:

> Standard attention: retain token-indexed historical representations for direct attention access.

> SSM-style representation: maintain an evolving compact state that carries information forward.

---

# 19. Scene H — Mamba-Style Selectivity

## Visual

Use a stream of tokens entering a state machine.

Each token receives a conceptual retention value:

```text
Token A → 0.95 KEEP
Token B → 0.20 COMPRESS
Token C → 0.03 DISCOUNT
Token D → 0.88 KEEP
```

Important tokens should have stronger visual influence on the state.

## Interaction

A token inspector should show:

```text
Token: "revenue"
Influence: high
State update: strong
```

versus:

```text
Token: "the"
Influence: low
State update: weak
```

Do not present these numbers as real model internals. Label them:

> **Educational selectivity visualization**

The goal is to teach the idea of input-dependent/selective state updates.

---

# 20. Scene I — Trade-Off Explorer

This is the most important interactive comparison screen.

## Layout

Left side:

```text
Choose architecture
○ Standard Attention
○ Sliding Window
○ Linear Attention
○ SSM
○ Mamba-style
○ Hybrid
```

Center:
3D representation of information flow.

Right side:
real-time metric cards.

Metrics:

```text
Memory pressure
Context handling
Direct retrieval capability
Latency tendency
Throughput tendency
```

For V1, use qualitative or normalized educational scores rather than fake precise benchmark numbers.

Example:

```text
Memory efficiency       ████████░░
Direct retrieval        █████░░░░░
Long-range state        ███████░░░
```

Add a label:

> **Conceptual comparison — not a benchmark**

---

# 21. Scene J — Hybrid Architecture Builder

This should be the final interactive experience.

## Core interaction

Give the user a vertical stack of layers.

Example:

```text
┌─────────────────┐
│ Attention       │
├─────────────────┤
│ Mamba-style     │
├─────────────────┤
│ Mamba-style     │
├─────────────────┤
│ Mamba-style     │
├─────────────────┤
│ Attention       │
├─────────────────┤
│ Mamba-style     │
└─────────────────┘
```

Users can:
- add a layer
- remove a layer
- change layer type
- reorder layers
- duplicate layers

## Real-time response

As architecture changes:

1. 3D information flow changes.
2. Cache representation changes.
3. Conceptual memory meter changes.
4. Retrieval/representation trade-off graph changes.
5. Layer statistics update.

## Critical wording

Do not say:

> "This exact ratio makes a better model."

Instead:

> "This configuration changes the balance between explicit attention and state-based processing. Different design choices can favor different memory, retrieval, and efficiency characteristics."

---

# 22. Hybrid Architecture Visualization

Use two distinct information paths.

```text
             INPUT TOKENS
                  ↓
          ┌───────┴───────┐
          ↓               ↓
     ATTENTION        STATE PATH
          ↓               ↓
   Token-level        Compact state
     retrieval         evolution
          └───────┬───────┘
                  ↓
               OUTPUT
```

Animate particles through the paths.

Attention path:
- show query-to-token connections
- allow a few connections to become visible on hover

State path:
- show tokens updating a compact state

This visual distinction is one of the core pedagogical moments of the product.

---

# 23. Real-Time Graphics Rules

Every user-controlled variable must have a corresponding visible consequence.

## Example

User changes:

```text
Context: 4K → 32K
```

Immediately update:
- token timeline scale
- cache blocks
- memory usage
- chart
- GPU occupancy visualization
- explanatory sentence

User changes:

```text
Users: 10 → 100
```

Immediately update:
- number of user lanes
- aggregate cache size
- memory pressure
- estimated concurrency status

User changes:

```text
Window: 128 → 16
```

Immediately update:
- active attention region
- evicted token visuals
- cache footprint
- old-token accessibility indicator

Do not use interactions that only change a number in a sidebar. The core graphic must respond.

---

# 24. Performance Requirements

The 3D experience must remain smooth on normal modern laptops.

## Rules

Do not create one heavy mesh per token for extremely long contexts.

Use:
- InstancedMesh
- GPU-friendly particles where appropriate
- object pooling
- level-of-detail representations
- aggregate token groups
- memoized geometry/materials
- throttled simulation calculations if needed

## Recommended visual scaling

```text
<= 200 tokens
Individual token objects

200–4,000 tokens
Instanced tokens / grouped bands

> 4,000 tokens
Aggregated token bands + sampled tokens + counters
```

The learner should feel they are looking at a large context without forcing the GPU to render every token individually.

## Animation performance

Use animation loops for the 3D scene only when necessary.

Prefer state-driven transitions for explanatory animations.

Avoid expensive post-processing effects as the default.

---

# 25. Responsive Design

Desktop-first because 3D interaction is the main experience.

Minimum target:
- 1280×720 usable experience

Also support:
- 1440p desktop
- laptop screens
- tablet as a reduced experience

On mobile:
- do not attempt full 3D complexity
- provide simplified 2D diagrams or reduced interaction if feasible

Mobile can be treated as secondary for V1.

---

# 26. UX Interaction Patterns

## Camera

Support:
- orbit
- zoom
- pan
- reset camera

Avoid completely free camera movement when it makes the explanatory object difficult to find.

Add a subtle "Reset view" button.

## Hover

Hover should reveal:
- token index
- token text
- whether K/V representation exists
- whether it is in active window
- state influence where relevant

## Click

Click should pin an explanation panel.

## Scroll

Use scroll only for moving between learning chapters, not for every micro interaction.

## Guided mode

A guided mode should automatically animate the key concepts.

A learner should be able to choose:

```text
[ Guided lesson ]
[ Explore freely ]
```

---

# 27. Guided Lesson Sequence

Recommended sequence:

### Chapter 1
**Where does the memory go?**

### Chapter 2
**KV Cache in 60 seconds**

### Chapter 3
**Watch the cache grow**

### Chapter 4
**Why long contexts hurt concurrency**

### Chapter 5
**Can we forget old tokens?**

### Chapter 6
**Can we compress history?**

### Chapter 7
**Can we carry history as state?**

### Chapter 8
**Can the state be selective?**

### Chapter 9
**Compare the trade-offs**

### Chapter 10
**Build a hybrid**

### Chapter 11
**Key takeaways**

---

# 28. Microcopy Guidelines

The website should use plain technical language.

Prefer:

> "More tokens mean more historical information to represent."

Instead of:

> "Token cardinality induces proportional expansion in autoregressive state residency."

Prefer:

> "The KV cache grows as the context grows."

Instead of:

> "Context-dependent memory footprint exhibits approximately linear asymptotic behavior."

Use short statements as visual anchors:

- **More context → more cache**
- **More users → more total cache**
- **Windowing → bounded recent context**
- **Compression → compact history**
- **State → evolving summary**
- **Selectivity → keep what matters**
- **Hybrid → combine strengths**

---

# 29. Equations / Reference Panel

The main experience should stay conceptual, but a "Under the hood" panel should expose technical detail.

Include:

### Simplified KV-cache growth

```text
KV bytes ≈
context tokens × layers × KV heads × head dimension × 2 × bytes/element
```

Explain the factor 2:

> One factor for K and one for V.

### Multi-user scaling

```text
Total KV memory ≈ per-user KV memory × active users
```

Add a note:

> This is an educational estimate. Real systems include additional memory consumers and implementation-specific effects.

### Sliding-window idea

```text
Accessible history ≈ recent W tokens
```

### State-based idea

```text
history → compact evolving state
```

Do not overwhelm the beginner with derivations in the main path.

---

# 30. Accuracy / Scientific Guardrails

The product must be technically honest.

Use terms such as:
- "conceptually"
- "simplified"
- "approximately"
- "educational simulation"

Where appropriate.

Examples of correct framing:

> "Linear attention refers to a family of formulations that avoid the same quadratic token-pair computation of standard softmax attention; implementations and exact properties vary."

> "SSMs represent sequence history through a state updated over time."

> "Mamba-style models introduce input-dependent/selective mechanisms in the state update."

Avoid presenting a single simplified diagram as a literal implementation detail of every model in the family.

---

# 31. Accessibility

All key information should also be available without relying on 3D graphics.

Requirements:
- keyboard-accessible controls
- readable labels
- sufficient text contrast
- reduced-motion option
- screen-readable summaries of numeric metrics
- 3D canvas has an accessible text explanation

Add a preference:

> **Reduce motion**

When enabled:
- no continuous particle animation
- use instant/short transitions
- preserve state changes and values

---

# 32. State Persistence

No account required for V1.

Optionally persist in localStorage:
- last chapter
- selected architecture
- last simulation settings
- reduced motion preference

This makes experimentation feel continuous.

---

# 33. Suggested Project Structure

```text
src/
├── app/
│   ├── routes/
│   │   ├── Home.tsx
│   │   ├── KvCache.tsx
│   │   ├── Playground.tsx
│   │   └── Reference.tsx
│   │
├── components/
│   ├── layout/
│   ├── controls/
│   ├── metrics/
│   ├── charts/
│   └── common/
│
├── three/
│   ├── scenes/
│   │   ├── KvCacheScene.tsx
│   │   ├── ContextGrowthScene.tsx
│   │   ├── SlidingWindowScene.tsx
│   │   ├── LinearAttentionScene.tsx
│   │   ├── SsmScene.tsx
│   │   ├── MambaScene.tsx
│   │   └── HybridScene.tsx
│   │
│   ├── objects/
│   │   ├── Token.tsx
│   │   ├── KeyValuePair.tsx
│   │   ├── CacheBlock.tsx
│   │   ├── StateNode.tsx
│   │   ├── AttentionLayer.tsx
│   │   └── GpuMemory.tsx
│   │
│   └── materials/
│
├── simulation/
│   ├── kvCache.ts
│   ├── serving.ts
│   ├── slidingWindow.ts
│   ├── linearAttention.ts
│   ├── stateSpace.ts
│   ├── mambaConcept.ts
│   ├── hybrid.ts
│   └── metrics.ts
│
├── state/
│   ├── useSimulationStore.ts
│   └── types.ts
│
├── data/
│   ├── lessons.ts
│   ├── defaults.ts
│   └── terminology.ts
│
└── styles/
    ├── tokens.css
    └── theme.css
```

Exact structure can change, but keep simulation, UI, and 3D presentation logically separate.

---

# 34. Data-Driven Lessons

Do not hardcode every lesson into UI components.

Use a data-driven lesson configuration.

Example:

```ts
interface LessonStep {
  id: string;
  title: string;
  objective: string;
  scene: SceneId;
  narration: string[];
  callouts?: Callout[];
  controls?: ControlConfig[];
  challenge?: ChallengeConfig;
}
```

This will make it much easier to add future topics later.

---

# 35. Educational Challenges

At least three interactive challenges should exist in V1.

## Challenge 1 — Fit users on a GPU

Prompt:

> "You have 24 GB GPU memory and 10 GB is occupied by model memory. Adjust context/user settings so the active users fit."

Success:

> "You stayed within the memory budget."

Failure:

> "The KV cache exceeded the remaining memory budget."

## Challenge 2 — Sliding window

Prompt:

> "Keep the recent conversation efficient, but choose a window large enough to retain a target token."

## Challenge 3 — Architecture builder

Prompt:

> "Create a configuration that prioritizes memory efficiency while keeping some explicit attention capacity."

Do not grade this as objectively correct. Grade it against stated preferences.

---

# 36. Final Summary Screen

End with a visual summary rather than a wall of text.

```text
             LONG CONTEXT
                  ↓
              KV CACHE
                  ↓
          MEMORY PRESSURE
                  ↓
       ┌──────────┼──────────┐
       ↓          ↓          ↓
   WINDOW      COMPRESS     STATE
       ↓          ↓          ↓
   Sliding     Linear       SSM
                             ↓
                         Selective
                             ↓
                           Mamba
       └──────────┬──────────┘
                  ↓
               HYBRIDS
```

Then a final statement:

> **The central engineering problem is not simply "make attention faster." It is deciding how much historical information to retain, how to represent it, and what trade-offs that representation creates.**

CTA:

> **Explore the architecture playground again**

Secondary CTA:

> **Open the technical reference**

---

# 37. What the MVP Must Demonstrate

The first working version is successful if a user can do all of the following:

1. Watch tokens produce K/V entries.
2. Watch the KV cache grow.
3. Increase context length and see memory increase.
4. Increase active users and see total memory increase.
5. Switch to sliding-window mode and watch old tokens leave the active window.
6. Switch to linear-attention mode and watch token history collapse into a state representation.
7. Switch to SSM mode and watch an evolving state carry history.
8. Switch to Mamba-style mode and see different tokens exert different conceptual influence.
9. Compare modes side-by-side.
10. Build a hybrid stack and see the visualization change.
11. Read the simplified technical equations.
12. Replay the guided lesson.

---

# 38. MVP Implementation Order

Do not attempt every visual simultaneously.

## Phase 1 — Foundation

Build:
- project scaffold
- design system
- routing
- simulation state
- central simulation engine
- basic Three.js scene

## Phase 2 — KV Cache

Build:
- token stream
- K/V objects
- cache container
- token generation animation
- memory calculation
- user lanes

This alone should already be a compelling mini-experience.

## Phase 3 — Sliding Window

Build:
- attention window
- token eviction
- window slider
- accessibility state

## Phase 4 — State / Compression Views

Build:
- linear-attention state visualization
- SSM state visualization
- Mamba-style selectivity visualization

## Phase 5 — Trade-off Explorer

Build:
- comparison metrics
- architecture switching
- synchronized animation

## Phase 6 — Hybrid Builder

Build:
- draggable/editable layers
- live architecture diagram
- live trade-off panel

## Phase 7 — Educational Polish

Build:
- guided narration
- challenges
- tooltips
- equations
- reduced motion
- responsive tuning

## Phase 8 — Performance / QA

Test:
- laptop GPU
- integrated graphics
- 60 FPS target for normal scenes where practical
- long-context symbolic rendering
- camera usability
- state reset behavior
- mobile fallback

---

# 39. Developer Acceptance Criteria

The implementation should not be considered complete merely because every page exists.

A feature is complete when:

### Visual correctness
- The 3D visualization clearly communicates the concept.
- No important object is hidden behind UI.
- Token flow direction is obvious.
- Active vs inactive/evicted history is obvious.

### Interaction correctness
- Controls produce immediate visible changes.
- Reset restores the entire state.
- No state drift between 3D and 2D metrics.

### Educational correctness
- Explanations match the simulated mechanism.
- Simplifications are explicitly labeled.
- The UI does not make unsupported claims about proprietary models.

### Technical quality
- TypeScript types are used throughout.
- Simulation logic is unit-testable without WebGL.
- 3D components are modular.
- Large token counts do not create one DOM/3D object per token.
- No unnecessary backend dependency.

### UX quality
- First meaningful visual appears quickly.
- Controls are self-explanatory.
- Users can always reset.
- User can enter guided mode and understand the story without external instruction.

---

# 40. Testing Strategy

## Unit tests

Test:
- KV memory formula
- multi-user aggregation
- sliding window behavior
- state-size assumptions
- architecture configuration parsing

## Integration tests

Verify:
- slider → simulation → metrics → 3D update
- architecture switch → correct scene
- reset → deterministic initial state

## Visual QA

Manually inspect:
- camera position
- labels
- token scaling
- memory meter
- animations
- overlapping UI
- high-context mode

## Performance tests

Test at:
- 10 tokens
- 100 tokens
- 1K tokens
- 4K tokens
- 16K simulated context
- 32K simulated context

Use aggregate rendering beyond the threshold described earlier.

---

# 41. Future Extension Hooks

Do not build these now, but make the codebase ready for them.

Potential future modules:

```text
AI SYSTEMS LAB
│
├── KV Cache & Token Compression   ← V1
├── Attention & Flash Attention
├── Prefill vs Decode
├── PagedAttention
├── Continuous Batching
├── Quantization
├── Speculative Decoding
├── MoE
├── GPU Memory Hierarchy
├── Multi-GPU Inference
└── End-to-End LLM Serving
```

Future modules should reuse:
- camera system
- design system
- simulation engine pattern
- lesson engine
- challenge engine
- metric cards
- 3D token objects
- animation utilities
- architecture graph components

---

# 42. Important Product Decision

Do not make the first release a "3D website for the sake of 3D."

Every 3D object must answer one of these questions:

- What information is moving?
- Where is it stored?
- What is being forgotten?
- What is being compressed?
- What state is evolving?
- What changes when the user changes a variable?

If an effect does not answer a learning question, remove it.

The 3D should make difficult systems concepts **visible**, not merely make the page look impressive.

---

# 43. Final Build Brief for Codex

Build Version 1 as a premium interactive educational web experience titled:

> **AI Systems Lab — Compressing Attention Across Tokens**

The first module must teach KV cache and token-history representation using a combination of:

- interactive 3D visualization
- real-time simulation
- transparent equations
- guided narration
- experiments
- trade-off exploration
- hybrid architecture construction

The learner should physically see the following transformation:

```text
Tokens
  ↓
Q / K / V
  ↓
KV Cache
  ↓
Context grows
  ↓
GPU memory pressure
  ↓
Alternative history representations
  ├── Sliding Window
  ├── Linear Attention
  ├── SSM
  └── Mamba-style selective state
  ↓
Trade-offs
  ↓
Hybrid Architecture
```

The product should feel like a **virtual laboratory for LLM inference concepts**.

Prioritize:

1. clarity
2. interaction
3. real-time cause-and-effect
4. technical honesty
5. visual polish
6. performance

Do not broaden the topic until the KV-cache/token-compression module is complete and genuinely useful on its own.

