import './style.css';
import { lessons } from './lessonData';
import { reading } from './lessonReading';
import { researchPapers, transformerLessons } from './transformerData';
import { attentionPaperChapters, attentionPaperEli5, upcomingPapers } from './researchData';

type Architecture = 'attention' | 'window' | 'linear' | 'ssm' | 'mamba' | 'hybrid';

type State = {
  tokens: number;
  users: number;
  window: number;
  architecture: Architecture;
  reducedMotion: boolean;
  chapter: number;
  eli5: boolean;
};

const architectureLabels: Record<Architecture, string> = {
  attention: 'Standard attention',
  window: 'Sliding window',
  linear: 'Linear attention',
  ssm: 'State-space model',
  mamba: 'Mamba-style selectivity',
  hybrid: 'Hybrid stack'
};

const state: State = { tokens: 1280, users: 12, window: 128, architecture: 'attention', reducedMotion: false, chapter: 0, eli5: false };
const model = { layers: 32, kvHeads: 8, headDim: 128, bytes: 2, gpu: 24, weights: 10 };
const root = document.querySelector<HTMLDivElement>('#app')!;
const transformerState = { lesson: 0, tokens: 7, heads: 4, depth: 12, path: 2, query: 4, eli5: false };
const transformerPath = ['Token embeddings', 'Position signal', 'Self-attention', 'MLP / residual update', 'Prediction head'];
const demoTokens = ['The', 'cat', 'sat', 'because', 'it', 'was', 'tired'];
const transformerPathDetails = [
  ['Token embeddings', 'Token IDs become learned vectors. This is where text enters the numerical model.', 'Input shape: sequence length × model width'],
  ['Position signal', 'Position information is combined with token content so the model can distinguish order.', 'Content + location → representation'],
  ['Self-attention', 'Each position compares a Query with Keys and mixes the matching Values from other positions.', 'Q, K, V → contextual representation'],
  ['MLP / residual update', 'The MLP transforms each position, while the residual path carries the previous representation forward.', 'representation + learned update'],
  ['Prediction head', 'The final representation is converted into scores over the vocabulary for the next-token decision.', 'hidden state → vocabulary logits']
];
const transformerEli5: Record<string, { question: string; intro: string; sections: { heading: string; body: string; bullets?: string[] }[]; takeaway: string }> = {
  tokens: { question: 'How does a Transformer see a sentence?', intro: 'Imagine giving the model a box of word labels. It turns every label into a small bundle of numbers so it can work with the sentence.', sections: [{ heading: 'Words become number cards', body: 'A tokenizer cuts a sentence into small pieces. Each piece gets a number, and an embedding turns that number into a learned number card.' }, { heading: 'Cards carry meaning clues', body: 'The card is not a dictionary definition. It is more like a location on a giant map where similar uses can end up near each other.' }, { heading: 'Why start here?', body: 'Everything else in the Transformer changes these number cards little by little.', bullets: ['Sentence → pieces.', 'Pieces → number cards.', 'Number cards → Transformer layers.'] }], takeaway: 'A Transformer starts by turning text pieces into number cards it can change.' },
  position: { question: 'Why is “dog bites man” different from “man bites dog”?', intro: 'Two sentences can use the same words but mean different things because the order changes. The model needs a little “where am I?” sticker for each token.', sections: [{ heading: 'Attention needs a map', body: 'Looking at the same collection of word cards is not enough. The model also needs to know which card came first, second, and third.' }, { heading: 'Add a place sticker', body: 'Position information travels with each token so the model can tell the difference between the words and their places.' }, { heading: 'Order changes meaning', body: 'Move the cards around and the model sees a different arrangement.', bullets: ['Content says what the token is.', 'Position says where it is.', 'Together they describe the sentence.'] }], takeaway: 'Position stickers help the model understand the order of the words.' },
  attention: { question: 'How can one word use information from the rest of the sentence?', intro: 'Imagine each word can look around the sentence and decide which other words are useful right now.', sections: [{ heading: 'A word asks a question', body: 'The Query is the word’s question: “What should I look for?”' }, { heading: 'Other words hold labels and messages', body: 'Keys are labels that help match the question. Values are the messages that get passed along when a match is useful.' }, { heading: 'Meaning becomes richer', body: 'After looking around, a word gets a new version of itself that includes helpful context.', bullets: ['Query asks.', 'Key matches.', 'Value carries the message.'] }], takeaway: 'Attention lets one word borrow useful clues from other words.' },
  block: { question: 'What happens after attention?', intro: 'After the words share clues, each word gets its own little thinking time to reshape what it learned.', sections: [{ heading: 'Share first', body: 'Attention lets the word cards talk to one another.' }, { heading: 'Think separately', body: 'An MLP is like a small workshop that changes each card on its own.' }, { heading: 'Keep the old notes', body: 'A residual connection keeps a copy of what came before, so each layer can add a helpful change instead of starting over.', bullets: ['Words share clues.', 'Each word is reshaped.', 'The old version is carried forward.'] }], takeaway: 'A Transformer block lets words share, think, and keep a running copy of their notes.' },
  heads: { question: 'Why not use one attention pattern?', intro: 'One pair of eyes may miss something. Multiple attention heads let the model look at the same sentence in several ways at once.', sections: [{ heading: 'Different pairs of eyes', body: 'One head might notice nearby words, while another notices a name and the word that refers to it.' }, { heading: 'Look, then combine', body: 'Each head makes its own small view. The model puts the views together afterward.' }, { heading: 'More is not always better', body: 'Extra heads are useful only when the model learns helpful jobs for them.', bullets: ['Each head sees a learned view.', 'Heads work at the same time.', 'Their views are combined.'] }], takeaway: 'Multiple heads give the model several ways to look for relationships.' },
  stack: { question: 'How does a simple operation become a capable model?', intro: 'One layer can make one small improvement. Many layers are like many rounds of editing a drawing until the picture becomes clear.', sections: [{ heading: 'Each layer adds a clue', body: 'Early layers may notice simple patterns. Later layers can use those clues to notice more complicated relationships.' }, { heading: 'The note travels upward', body: 'Residual connections carry the working version through the stack while each layer adds its update.' }, { heading: 'Make a guess', body: 'At the end, the model turns its final notes into scores for possible next words.', bullets: ['Layer 1 makes a change.', 'Layer 2 builds on it.', 'The final layer helps choose the next token.'] }], takeaway: 'Many small rounds of sharing and editing can produce a useful next-word prediction.' }
};
const kvEli5: Record<string, { question: string; explanation: string; intro: string; sections: { heading: string; body: string; formula?: string; bullets?: string[] }[]; analogy: string; visual: string; tradeoff: string; takeaway: string }> = {
  'kv-cache': { question: 'How can the model remember earlier words?', explanation: 'Imagine a helper keeping little information cards on a shelf. The model can look at those cards instead of making them again.', intro: 'A KV cache is a shelf where the model keeps useful cards from earlier words.', sections: [{ heading: 'Question cards', body: 'A Query is like asking, “What do I need?” A Key is like a label on a card. A Value is the message written on the card.' }, { heading: 'Keep the cards ready', body: 'When a new word arrives, the model can reuse the old cards instead of making every card again.' }, { heading: 'The shelf gets bigger', body: 'Every new word can add another pair of cards, so a long conversation needs more shelf space.', bullets: ['The cards are not the whole model.', 'The cards are saved work.', 'Saved work uses memory.'] }], analogy: 'A librarian keeps labeled cards ready for the next question.', visual: 'Words become pairs of reusable cards.', tradeoff: 'Saving work is faster, but the shelf takes up space.', takeaway: 'The KV cache is a shelf of saved Key and Value cards.' },
  growth: { question: 'What happens when the conversation gets longer?', explanation: 'More words mean more cards. More people using the model means more shelves.', intro: 'KV memory grows because every remembered word needs room for its cards.', sections: [{ heading: 'One more word', body: 'Each extra word can add another Key and Value pair. Double the remembered words and the card storage roughly doubles.' }, { heading: 'Many people', body: 'Each active conversation has its own shelf. A busy server must hold all of them at once.' }, { heading: 'The important idea', body: 'The model weights already use memory. Conversation shelves use additional memory.', bullets: ['Longer chat → more cards.', 'More users → more shelves.', 'More shelves → fewer users fit.'] }], analogy: 'A small backpack is fine for a short trip; many long trips need a much bigger storage room.', visual: 'The cache grows as the word count and user count grow.', tradeoff: 'Remembering more can help, but it costs space.', takeaway: 'Long context creates memory pressure because every remembered word needs storage.' },
  window: { question: 'Can the model keep only the newest words?', explanation: 'A sliding window is like looking through a small window at the newest part of a long conversation.', intro: 'The model keeps a bounded recent area instead of looking directly at every old word.', sections: [{ heading: 'The window moves', body: 'When new words arrive, the window slides forward. Old words fall outside the view.' }, { heading: 'Small window', body: 'A small window needs less shelf space, but an old secret may be outside the view.' }, { heading: 'A careful claim', body: 'This does not mean the whole model has no long memory. Other layers or mechanisms may still carry information.', bullets: ['Recent words stay visible.', 'Old words become less direct.', 'The window size controls the trade-off.'] }], analogy: 'Read only the last few lines of a very long note.', visual: 'A bright window moves along the word strip.', tradeoff: 'Smaller view, smaller memory; bigger view, more direct history.', takeaway: 'Sliding-window attention saves space by looking directly at only a recent region.' },
  linear: { question: 'Can many old words become one small summary?', explanation: 'Instead of keeping every card open, the model can keep a running notebook that changes as new words arrive.', intro: 'Linear-attention methods try to accumulate the past into a more compact running representation.', sections: [{ heading: 'A running notebook', body: 'Each new word updates the notebook. The notebook carries something about the past forward.' }, { heading: 'Why it can help', body: 'A small notebook may use less space than a card for every word.' }, { heading: 'What can be difficult', body: 'A short notebook may not remember every exact detail as directly as a full card shelf.', bullets: ['This is a family of methods.', 'It changes how history is processed.', 'The simulator is a teaching picture, not a benchmark.'] }], analogy: 'Write a growing summary instead of carrying every page of a book.', visual: 'Many word cards flow into a compact state.', tradeoff: 'Small summary, harder exact lookup.', takeaway: 'Linear attention tries to carry history in a more compact running form.' },
  ssm: { question: 'What if memory is a changing state?', explanation: 'An SSM is like a person reading a story and keeping an evolving idea of what is happening.', intro: 'A state-space model carries information forward in a hidden state that changes with each new word.', sections: [{ heading: 'The state changes', body: 'A new word arrives, and the current state is updated. The next word sees the new state.' }, { heading: 'Not a card shelf', body: 'The state is a different way to represent history; it is not simply a smaller copy of attention cards.' }, { heading: 'Finite room', body: 'A compact state must decide what useful information can continue forward.', bullets: ['Read a word.', 'Update the state.', 'Use the state for the next word.'] }], analogy: 'Keep a changing summary of a story in your head.', visual: 'Word 1 changes State 1, then Word 2 changes State 2.', tradeoff: 'Compact memory, finite detail.', takeaway: 'SSMs carry history through an evolving state.' },
  mamba: { question: 'Should every word change memory equally?', explanation: 'A useful word may deserve a strong note, while a tiny formatting word may need only a small note.', intro: 'Mamba-style selectivity lets state updates depend on the incoming word.', sections: [{ heading: 'Choose what matters', body: 'The model can process different words differently instead of treating every word exactly the same.' }, { heading: 'A teaching picture', body: 'A fact such as a price might make a strong state update. A filler word might make a smaller one.' }, { heading: 'Not a human score', body: 'The activity’s strength values are teaching symbols, not literal internal importance numbers.', bullets: ['Input changes the update.', 'Some information can be retained more strongly.', 'The actual behavior is learned.'] }], analogy: 'Underline an important sentence and skim a repeated filler phrase.', visual: 'Different words make different-sized state updates.', tradeoff: 'Spend memory on useful information, but do not assume a perfect importance judge.', takeaway: 'Mamba-style models make state updates input-dependent and selective.' },
  tradeoffs: { question: 'Which kind of memory is useful here?', explanation: 'A full archive is easy to search but heavy to carry. A small notebook is light but may not contain every exact detail.', intro: 'There is no single best way to remember every kind of conversation.', sections: [{ heading: 'Precise or compact?', body: 'Token cards are direct and detailed. States and summaries are compact but may hide some exact details.' }, { heading: 'The work changes the answer', body: 'A chatbot, a long document reader, and a high-throughput server may prefer different balances.' }, { heading: 'Numbers need labels', body: 'The meters in this lab are conceptual teaching aids, not universal scores.', bullets: ['Memory matters.', 'Retrieval matters.', 'Speed and workload matter too.'] }], analogy: 'Choose between a searchable filing cabinet and a small notebook.', visual: 'The comparison changes when you choose a different memory style.', tradeoff: 'Every design balances useful properties.', takeaway: 'Architecture is a set of trade-offs, not a universal ranking.' },
  hybrid: { question: 'Can we use more than one kind of memory?', explanation: 'A hybrid uses different tools in different layers: one tool for precise looking and another for carrying a compact story forward.', intro: 'Hybrid architectures combine attention and state-based processing.', sections: [{ heading: 'Two tools', body: 'Attention can look directly at token relationships. State-based layers can carry a compact evolving representation.' }, { heading: 'Different places', body: 'A model can place these layers in different positions and proportions.' }, { heading: 'No magic ratio', body: 'There is no universal recipe that guarantees a better model. Training, hardware, data, and workload matter too.', bullets: ['Attention path: direct lookup.', 'State path: compact history.', 'Hybrid path: combine tools.'] }], analogy: 'Use a magnifying glass for tiny details and a backpack for carrying the whole trip.', visual: 'The input can travel through attention and state paths.', tradeoff: 'Combining tools may balance strengths, but it also adds design choices.', takeaway: 'Hybrid models combine different ways of handling history.' }
};

function formatBytes(gb: number): string {
  return gb >= 1 ? `${gb.toFixed(2)} GB` : `${(gb * 1024).toFixed(0)} MB`;
}

function kvGb(tokens: number): number {
  return tokens * model.layers * model.kvHeads * model.headDim * model.bytes * 2 / (1024 ** 3);
}

function effectiveMemory(): number {
  if (state.architecture === 'window') return kvGb(Math.min(state.tokens, state.window));
  if (state.architecture === 'linear' || state.architecture === 'ssm') return kvGb(Math.min(state.tokens, 256)) * 0.42;
  if (state.architecture === 'mamba') return kvGb(Math.min(state.tokens, 192)) * 0.3;
  if (state.architecture === 'hybrid') return kvGb(Math.min(state.tokens, state.window * 2)) * 0.55;
  return kvGb(state.tokens);
}

function metrics() {
  const perUser = effectiveMemory();
  const total = perUser * state.users;
  const available = model.gpu - model.weights;
  const fit = Math.max(0, Math.floor(available / Math.max(perUser, 0.0001)));
  const retrieval = { attention: 98, window: Math.min(96, 48 + state.window / 4), linear: 63, ssm: 58, mamba: 64, hybrid: 82 }[state.architecture];
  const bounded = state.architecture === 'attention' ? 12 : state.architecture === 'window' ? 72 : state.architecture === 'hybrid' ? 58 : 88;
  return { perUser, total, available, fit, retrieval, bounded, pressure: Math.min(100, (total / available) * 100) };
}

function render() {
  if (window.location.hash === '#research') {
    renderResearch();
    return;
  }
  if (window.location.hash === '#transformer') {
    renderTransformer();
    return;
  }
  const lesson = lessons[state.chapter];
  const lessonReading = reading[lesson.id];
  const lessonCopy = state.eli5 ? kvEli5[lesson.id] : { question: lesson.question, explanation: lesson.explanation, intro: lessonReading.intro, sections: lessonReading.sections, analogy: lesson.analogy, visual: lesson.visual, tradeoff: lesson.tradeoff, takeaway: lesson.takeaway };
  const m = metrics();
  root.innerHTML = `
    <div id="hover-card" class="hover-card" role="status" aria-live="polite"><span class="hover-card-kicker">CONTEXT NOTE</span><strong data-card-title></strong><p data-card-body></p></div>
    <header class="topbar">
      <a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a>
      <div class="topbar-meta"><a href="#transformer">MODULE 01 / TRANSFORMER</a><span class="status-dot"></span><a href="#top">MODULE 02 / KV CACHE</a><a href="#research" target="_blank" rel="noreferrer">RESEARCH PAPERS ↗</a></div>
      <button class="quiet-button" data-action="motion">${state.reducedMotion ? 'Motion off' : 'Reduce motion'}</button>
    </header>
    <main id="top">
      <section class="hero section-shell">
        <div class="hero-copy">
          <p class="eyebrow">Interactive technical laboratory</p>
          <h1>Compressing attention <em>across tokens.</em></h1>
          <p class="hero-lede">Long context is not free. Explore how historical tokens occupy memory, then experiment with the architectures that represent history differently.</p>
          <div class="hero-actions"><button class="primary-button" data-action="start">Enter the lab <span>↗</span></button><button class="text-button" data-action="reference">Read the equation <span>↓</span></button></div>
        </div>
        <div class="hero-signal info-target" tabindex="0" data-info-title="Context becomes residency" data-info-body="Every token adds a Key and Value pair to the standard attention cache. The cache remains available for later decoding steps." aria-label="A visual showing tokens becoming memory pressure">
          <div class="signal-label"><span>CONTEXT</span><strong>${state.tokens.toLocaleString()} tokens</strong></div>
          <div class="signal-lines"><span></span><span></span><span></span><span></span><span></span></div>
          <div class="signal-cache"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
          <div class="signal-foot"><span>Q / K / V</span><strong>${formatBytes(m.perUser)} / user</strong></div>
        </div>
      </section>

      <section class="lab-section section-shell" id="lab">
        <div class="lesson-layout"><aside class="lesson-sidebar"><div class="sidebar-heading"><span class="lesson-label">MODULE MAP</span><strong>Choose a lesson</strong></div><nav aria-label="Lesson navigation">${lessons.map((item, index) => `<button class="lesson-nav-item ${index === state.chapter ? 'active' : ''}" data-chapter="${index}"><span>${item.kicker}</span><strong>${item.title}</strong><small>${item.question}</small></button>`).join('')}</nav></aside><div class="lesson-content">
        <div class="section-heading"><div><p class="eyebrow">${lesson.kicker}</p><h2>${lesson.title}</h2></div><div class="lesson-step-actions"><button class="step-button" data-action="previousLesson" ${state.chapter === 0 ? 'disabled' : ''}>← Previous</button><span>${state.chapter + 1} / ${lessons.length}</span><button class="step-button" data-action="nextLesson" ${state.chapter === lessons.length - 1 ? 'disabled' : ''}>Next →</button></div></div>
        <p class="section-intro">${lessonCopy.question} ${lessonCopy.explanation}</p>
        <article class="reading-panel"><div class="reading-header"><div><span class="lesson-label">${state.eli5 ? 'ELI5 READING' : 'READ THE CONCEPT'}</span><h3>${lesson.title}</h3></div><span class="reading-index">${String(state.chapter + 1).padStart(2, '0')} / ${String(lessons.length).padStart(2, '0')}</span></div><p class="reading-intro">${lessonCopy.intro}</p><div class="reading-sections">${lessonCopy.sections.map((section) => `<section class="reading-section"><h4>${section.heading}</h4><p>${section.body}</p>${section.formula ? `<code>${section.formula}</code>` : ''}${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>` : ''}</section>`).join('')}</div></article>
        <div class="lesson-strip"><div class="lesson-column"><span class="lesson-label">INTUITION</span><p>${lessonCopy.analogy}</p></div><div class="lesson-column"><span class="lesson-label">WHAT TO WATCH</span><p>${lessonCopy.visual}</p></div><div class="lesson-column"><span class="lesson-label">TRADE-OFF</span><p>${lessonCopy.tradeoff}</p></div></div>
        <div class="lab-grid">
          <div class="visual-panel info-target" tabindex="0" data-info-title="Token history representation" data-info-body="The canvas is a symbolic view. It samples long contexts so the browser does not create thousands of heavy objects, while the formulas still use the full token count.">
            <div class="panel-topline"><span class="panel-label">LIVE MEMORY MODEL</span><span class="panel-note">Simplified estimate · not a benchmark</span></div>
            <canvas id="lab-canvas" aria-label="Animated diagram of token history and memory representation"></canvas>
            <div class="canvas-legend"><span><b class="dot q"></b>Query</span><span><b class="dot k"></b>Key</span><span><b class="dot v"></b>Value</span><span><b class="dot state-dot-legend"></b>State</span></div>
          </div>
          <aside class="control-panel">
            <div class="control-block info-target" tabindex="0" data-info-title="Context length" data-info-body="Standard attention stores history per token, so this is the strongest driver of per-user KV memory."><label for="tokens"><span>Context length</span><strong>${state.tokens.toLocaleString()}</strong></label><input id="tokens" type="range" min="10" max="32768" step="10" value="${state.tokens}"><div class="range-labels"><span>10</span><span>32K</span></div></div>
            <div class="control-block info-target" tabindex="0" data-info-title="Active users" data-info-body="Each user brings a separate sequence cache. Total active KV memory is approximately per-user KV memory multiplied by users."><label for="users"><span>Active users</span><strong>${state.users}</strong></label><input id="users" type="range" min="1" max="100" value="${state.users}"><div class="range-labels"><span>1</span><span>100</span></div></div>
            <div class="control-block info-target" tabindex="0" data-info-title="Window size" data-info-body="Sliding-window attention keeps only the most recent W token positions directly available to attention."><label for="window"><span>Window size</span><strong>${state.window}</strong></label><input id="window" type="range" min="16" max="1024" step="16" value="${state.window}"><div class="range-labels"><span>16</span><span>1K</span></div></div>
            <div class="architecture-select"><span class="control-caption">History representation</span><div class="architecture-grid">${(Object.keys(architectureLabels) as Architecture[]).map((key) => `<button class="arch-button ${key === state.architecture ? 'selected' : ''}" data-architecture="${key}">${architectureLabels[key]}</button>`).join('')}</div></div>
          </aside>
        </div>
        <div class="metrics-grid" aria-live="polite">
          <div class="metric-card featured info-target" tabindex="0" data-info-title="KV memory per user" data-info-body="This estimate counts the Key and Value tensors for one sequence using 32 layers, 8 KV heads, 128-wide heads, and two bytes per element."><span>KV memory / user</span><strong>${formatBytes(m.perUser)}</strong><small>${state.architecture === 'attention' ? 'grows with every token' : 'bounded in this model'}</small></div>
          <div class="metric-card info-target" tabindex="0" data-info-title="Total active KV" data-info-body="This is the estimated cache held for all active users. Model weights are not included in this number."><span>Total active KV</span><strong>${formatBytes(m.total)}</strong><small>${state.users} users × per-user cache</small></div>
          <div class="metric-card info-target" tabindex="0" data-info-title="Users within budget" data-info-body="A simple capacity estimate: available GPU memory divided by the current per-user cache. Real serving systems also account for scheduling and other buffers."><span>Users within budget</span><strong>${m.fit}</strong><small>of ${state.users} requested</small></div>
          <div class="metric-card info-target" tabindex="0" data-info-title="Direct retrieval" data-info-body="A conceptual teaching score for how directly the architecture can address token-level history. It is not a benchmark or a quality claim."><span>Direct retrieval</span><strong>${Math.round(m.retrieval)}<small>/100</small></strong><small>conceptual score</small></div>
        </div>
        <div class="takeaway-grid"><div class="takeaway-card"><span class="lesson-label">ONE-SENTENCE TAKEAWAY</span><strong>${lessonCopy.takeaway}</strong></div><details class="check-card"><summary>Check your understanding</summary><p>${state.eli5 ? `In simple words: ${lessonCopy.question}` : lesson.check.prompt}</p><span>${state.eli5 ? lessonCopy.takeaway : lesson.check.answer}</span></details></div>
        </div></div>
      </section>

      <section class="explainer section-shell">
        <div class="explainer-copy"><p class="eyebrow">The engineering question</p><h2>Do we need to represent every old token in exactly the same way?</h2><p>Standard attention keeps token-indexed Key and Value records for direct lookup. Other designs trade some form of direct access for a smaller or bounded representation of history.</p></div>
        <div class="paths"><div class="path-row info-target" tabindex="0" data-info-title="Sliding window" data-info-body="Forget older direct attention records and keep a bounded recent region. Efficient, but distant information is no longer directly accessible."><span>FORGET</span><strong>Window</strong><small>Keep recent context</small></div><div class="path-row info-target" tabindex="0" data-info-title="Linear attention" data-info-body="Accumulate information into a running representation instead of retaining the same token-by-token cache form. Exact formulations vary."><span>COMPRESS</span><strong>Linear</strong><small>Accumulate a running state</small></div><div class="path-row info-target" tabindex="0" data-info-title="State-space model" data-info-body="Update an evolving state as tokens arrive. The state carries sequence history forward without exposing every old token as a separate lookup record."><span>EVOLVE</span><strong>SSM</strong><small>Carry history forward</small></div><div class="path-row info-target" tabindex="0" data-info-title="Mamba-style selectivity" data-info-body="Use input-dependent state updates so different tokens can have different influence on what is retained. This is an educational abstraction."><span>SELECT</span><strong>Mamba-style</strong><small>Update state selectively</small></div></div>
      </section>

      <section class="reference section-shell" id="reference"><div><p class="eyebrow">Under the hood</p><h2>The cache estimate stays transparent.</h2><p>These numbers are educational approximations. Model weights, runtime buffers, fragmentation, and implementation details also consume memory in real systems.</p></div><div class="equation-card info-target" tabindex="0" data-info-title="Why the factor of two?" data-info-body="The simplified estimate stores both a Key and a Value tensor for each token, so the formula includes one factor for each representation."><span>SIMPLIFIED KV CACHE GROWTH</span><code>KV bytes ≈ context × layers × KV heads<br>× head dimension × 2 × bytes / element</code><div class="equation-result"><span>${state.tokens.toLocaleString()} × ${model.layers} × ${model.kvHeads} × ${model.headDim} × 2 × ${model.bytes}</span><strong>${formatBytes(kvGb(state.tokens))}</strong></div></div></section>
    </main>
    <footer class="footer section-shell"><span>AI SYSTEMS LAB / 2026</span><span>One module. Many ways to remember.</span><button class="text-button" data-action="reset">Reset experiment</button></footer>
  `;
  bindEvents();
  bindInfoCards();
  drawCanvas();
}

function bindEvents() {
  const sidebarHeading = document.querySelector<HTMLElement>('#lab .sidebar-heading');
  if (sidebarHeading && !sidebarHeading.querySelector('[data-action="toggleEli5"]')) {
    const toggle = document.createElement('button');
    toggle.className = `eli5-toggle ${state.eli5 ? 'active' : ''}`;
    toggle.dataset.action = 'toggleEli5';
    toggle.setAttribute('aria-pressed', String(state.eli5));
    toggle.innerHTML = `ELI5 <span>${state.eli5 ? 'ON' : 'OFF'}</span>`;
    sidebarHeading.appendChild(toggle);
  }
  document.querySelector<HTMLInputElement>('#tokens')?.addEventListener('input', (event) => { state.tokens = Number((event.target as HTMLInputElement).value); render(); });
  document.querySelector<HTMLInputElement>('#users')?.addEventListener('input', (event) => { state.users = Number((event.target as HTMLInputElement).value); render(); });
  document.querySelector<HTMLInputElement>('#window')?.addEventListener('input', (event) => { state.window = Number((event.target as HTMLInputElement).value); render(); });
  document.querySelectorAll<HTMLButtonElement>('[data-architecture]').forEach((button) => button.addEventListener('click', () => { state.architecture = button.dataset.architecture as Architecture; render(); }));
  document.querySelectorAll<HTMLButtonElement>('[data-chapter]').forEach((button) => button.addEventListener('click', () => { state.chapter = Number(button.dataset.chapter); render(); }));
  document.querySelector<HTMLButtonElement>('[data-action="previousLesson"]')?.addEventListener('click', () => changeLesson(-1));
  document.querySelector<HTMLButtonElement>('[data-action="nextLesson"]')?.addEventListener('click', () => changeLesson(1));
  document.querySelector<HTMLButtonElement>('[data-action="start"]')?.addEventListener('click', () => document.querySelector('#lab')?.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth' }));
  document.querySelector<HTMLButtonElement>('[data-action="reference"]')?.addEventListener('click', () => document.querySelector('#reference')?.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth' }));
  document.querySelector<HTMLButtonElement>('[data-action="motion"]')?.addEventListener('click', () => { state.reducedMotion = !state.reducedMotion; render(); });
  document.querySelector<HTMLButtonElement>('[data-action="toggleEli5"]')?.addEventListener('click', () => { state.eli5 = !state.eli5; render(); });
  document.querySelector<HTMLButtonElement>('[data-action="reset"]')?.addEventListener('click', () => { Object.assign(state, { tokens: 1280, users: 12, window: 128, architecture: 'attention', chapter: 0 }); render(); window.scrollTo({ top: 0, behavior: 'auto' }); });
}

function changeLesson(direction: number) {
  state.chapter = Math.max(0, Math.min(lessons.length - 1, state.chapter + direction));
  render();
  document.querySelector('#lab')?.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth', block: 'start' });
}

function bindInfoCards() {
  const card = document.querySelector<HTMLDivElement>('#hover-card');
  if (!card) return;
  document.querySelectorAll<HTMLElement>('.info-target').forEach((target) => {
    const show = () => {
      const bounds = target.getBoundingClientRect();
      card.querySelector<HTMLElement>('[data-card-title]')!.textContent = target.dataset.infoTitle ?? '';
      card.querySelector<HTMLElement>('[data-card-body]')!.textContent = target.dataset.infoBody ?? '';
      card.style.left = `${Math.min(Math.max(18, bounds.left), window.innerWidth - 338)}px`;
      card.style.top = `${Math.min(Math.max(18, bounds.bottom + 12), window.innerHeight - card.offsetHeight - 18)}px`;
      card.classList.add('visible');
    };
    const hide = () => card.classList.remove('visible');
    target.addEventListener('mouseenter', show);
    target.addEventListener('mouseleave', hide);
    target.addEventListener('focus', show);
    target.addEventListener('blur', hide);
  });
}

function renderResearch() {
  const chapter = attentionPaperChapters[state.chapter % attentionPaperChapters.length];
    const researchBody = state.eli5 ? attentionPaperEli5[state.chapter % attentionPaperEli5.length][1] : chapter.body;
  const paper = researchPapers[0];
  root.innerHTML = `
    <header class="topbar"><a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a><div class="topbar-meta"><a href="#transformer">MODULE 01 / TRANSFORMER</a><span class="status-dot"></span><a href="#top">MODULE 02 / KV CACHE</a><a href="#research">RESEARCH PAPERS</a></div><button class="quiet-button" data-action="motion">${state.reducedMotion ? 'Motion off' : 'Reduce motion'}</button></header>
    <main id="research-top"><section class="research-hero section-shell"><div><p class="eyebrow">Research paper 01 / beginner track</p><h1>Attention Is <em>All You Need.</em></h1><p class="hero-lede">A friendly walk through the paper that introduced the Transformer. No assumed research background: first understand the problem, then see the idea, then connect it to the systems you use today.</p><div class="paper-meta research-hero-meta"><span>${paper.year}</span><span>${paper.citation}</span><span>Vaswani et al.</span></div></div><div class="research-thesis"><span class="lesson-label">THE PAPER IN ONE LINE</span><strong>Let every word ask which other words it should listen to.</strong><div class="research-thesis-flow"><span>tokens</span><b>→</b><span>attention</span><b>→</b><span>context</span></div></div></section>
    <section class="paper-reader section-shell"><aside class="paper-reader-nav"><div class="sidebar-heading"><span class="lesson-label">PAPER MAP</span><strong>Read the argument</strong></div><nav aria-label="Research paper chapters">${attentionPaperChapters.map((item, index) => `<button class="paper-chapter ${index === state.chapter % attentionPaperChapters.length ? 'active' : ''}" data-research-chapter="${index}"><span>${String(index + 1).padStart(2, '0')}</span><strong>${item.title}</strong><small>${item.label}</small></button>`).join('')}</nav></aside><article class="paper-reading"><div class="paper-reading-header"><div><span class="lesson-label">${chapter.label}</span><h2>${chapter.title}</h2></div><div class="lesson-step-actions"><button class="step-button" data-action="previousResearch" ${state.chapter === 0 ? 'disabled' : ''}>← Previous</button><span>${state.chapter + 1} / ${attentionPaperChapters.length}</span><button class="step-button" data-action="nextResearch" ${state.chapter === attentionPaperChapters.length - 1 ? 'disabled' : ''}>Next →</button></div></div><p class="paper-reading-body">${chapter.body}</p>${chapter.formula ? `<code class="paper-formula">${chapter.formula}</code>` : ''}<div class="paper-takeaway"><span class="lesson-label">KEEP THIS IDEA</span><strong>${chapter.takeaway}</strong></div><div class="paper-activity"><span class="lesson-label">LOOK AT THE SYSTEM</span><p>${chapter.label === 'The mechanism' ? 'The Q / K / V path is the same idea that later creates the KV cache. Move to the KV-cache module after this chapter to see the serving cost.' : chapter.label === 'Why it mattered' ? 'The paper’s idea is architectural: make relationships between positions easy to compute in parallel. The exact model and training recipe can change around it.' : 'Read the chapter, then use the Transformer module to manipulate the corresponding stage.'}</p><a href="${chapter.label === 'The mechanism' ? '#top' : '#transformer'}">Open the related interactive module →</a></div></article></section>
    <section class="upcoming-section section-shell"><div class="paper-heading"><div><p class="eyebrow">Research shelf</p><h2>More papers, coming soon.</h2><p>Each paper will get the same treatment: the problem, the idea, the mechanism, the evidence, and the engineering consequences.</p></div><span class="paper-count">${upcomingPapers.length} IN QUEUE</span></div><div class="upcoming-grid">${upcomingPapers.map(([title, description], index) => `<article class="upcoming-card"><span>COMING SOON · 0${index + 2}</span><h3>${title}</h3><p>${description}</p></article>`).join('')}</div></section></main><footer class="footer section-shell"><span>AI SYSTEMS LAB / RESEARCH PAPERS</span><a class="text-button" href="#transformer">Back to Transformer →</a></footer>`;
  bindResearchEvents();
}

function bindResearchEvents() {
  const currentChapter = attentionPaperChapters[state.chapter % attentionPaperChapters.length];
  const body = document.querySelector<HTMLElement>('.paper-reading-body');
  const takeaway = document.querySelector<HTMLElement>('.paper-takeaway strong');
  if (body) body.textContent = state.eli5 ? attentionPaperEli5[state.chapter % attentionPaperEli5.length][1] : currentChapter.body;
  if (takeaway) takeaway.textContent = state.eli5 ? 'Think of it as giving every word a chance to look around for helpful clues.' : currentChapter.takeaway;
  const sidebarHeading = document.querySelector<HTMLElement>('.paper-reader-nav .sidebar-heading');
  if (sidebarHeading && !sidebarHeading.querySelector('[data-action="toggleEli5"]')) {
    const toggle = document.createElement('button');
    toggle.className = `eli5-toggle ${state.eli5 ? 'active' : ''}`;
    toggle.dataset.action = 'toggleEli5';
    toggle.setAttribute('aria-pressed', String(state.eli5));
    toggle.innerHTML = `ELI5 <span>${state.eli5 ? 'ON' : 'OFF'}</span>`;
    sidebarHeading.appendChild(toggle);
  }
  document.querySelectorAll<HTMLButtonElement>('[data-research-chapter]').forEach((button) => button.addEventListener('click', () => { state.chapter = Number(button.dataset.researchChapter); renderResearch(); }));
  document.querySelector<HTMLButtonElement>('[data-action="previousResearch"]')?.addEventListener('click', () => { state.chapter = Math.max(0, state.chapter - 1); renderResearch(); });
  document.querySelector<HTMLButtonElement>('[data-action="nextResearch"]')?.addEventListener('click', () => { state.chapter = Math.min(attentionPaperChapters.length - 1, state.chapter + 1); renderResearch(); });
  document.querySelector<HTMLButtonElement>('[data-action="motion"]')?.addEventListener('click', () => { state.reducedMotion = !state.reducedMotion; renderResearch(); });
  document.querySelector<HTMLButtonElement>('[data-action="toggleEli5"]')?.addEventListener('click', () => { state.eli5 = !state.eli5; renderResearch(); });
}

function renderTransformer() {
  const lesson = transformerLessons[transformerState.lesson];
  const copy = transformerState.eli5 ? transformerEli5[lesson.id] : lesson;
  const paper = researchPapers[0];
  root.innerHTML = `
    <header class="topbar"><a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a><div class="topbar-meta"><a href="#transformer">MODULE 01 / TRANSFORMER</a><span class="status-dot"></span><a href="#top">MODULE 02 / KV CACHE</a><a href="#research" target="_blank" rel="noreferrer">RESEARCH PAPERS ↗</a></div><button class="quiet-button" data-action="motion">${state.reducedMotion ? 'Motion off' : 'Reduce motion'}</button></header>
    <main id="transformer-top"><section class="hero section-shell transformer-hero"><div class="hero-copy"><p class="eyebrow">Module 01 / Foundations</p><h1>How a Transformer <em>thinks in layers.</em></h1><p class="hero-lede">Follow a sequence from token IDs to contextual representations. See where attention connects positions, where the MLP transforms features, and why many blocks are stacked together.</p><div class="hero-actions"><button class="primary-button" data-action="startTransformer">Start the walkthrough <span>↗</span></button><a class="text-button" href="#top">Explore KV cache <span>→</span></a></div></div><div class="transformer-hero-diagram"><div class="diagram-caption">A SEQUENCE BECOMES A REPRESENTATION</div><div class="hero-token-row"><span>the</span><span>model</span><span>reads</span><span>context</span></div><div class="hero-arrow">↓</div><div class="hero-layer-row"><b>ATTENTION</b><b>MLP</b><b>ATTENTION</b></div><div class="hero-arrow">↓</div><div class="hero-output">contextual prediction</div></div></section>
    <section class="lab-section section-shell transformer-section" id="transformer-lab"><div class="lesson-layout"><aside class="lesson-sidebar"><div class="sidebar-heading"><span class="lesson-label">MODULE MAP</span><strong>Transformer path</strong></div><nav aria-label="Transformer lesson navigation">${transformerLessons.map((item, index) => `<button class="lesson-nav-item ${index === transformerState.lesson ? 'active' : ''}" data-transformer-lesson="${index}"><span>${item.kicker}</span><strong>${item.title}</strong><small>${item.question}</small></button>`).join('')}</nav></aside><div class="lesson-content"><div class="section-heading"><div><p class="eyebrow">${lesson.kicker}</p><h2>${lesson.title}</h2></div><div class="lesson-step-actions"><button class="step-button" data-action="previousTransformer" ${transformerState.lesson === 0 ? 'disabled' : ''}>← Previous</button><span>${transformerState.lesson + 1} / ${transformerLessons.length}</span><button class="step-button" data-action="nextTransformer" ${transformerState.lesson === transformerLessons.length - 1 ? 'disabled' : ''}>Next →</button></div></div><p class="section-intro">${lesson.question} ${lesson.intro}</p><article class="reading-panel"><div class="reading-header"><div><span class="lesson-label">READ THE CONCEPT</span><h3>${lesson.title}</h3></div><span class="reading-index">${String(transformerState.lesson + 1).padStart(2, '0')} / ${String(transformerLessons.length).padStart(2, '0')}</span></div><p class="reading-intro">${lesson.intro}</p><div class="reading-sections">${lesson.sections.map((section) => `<section class="reading-section"><h4>${section.heading}</h4><p>${section.body}</p>${section.formula ? `<code>${section.formula}</code>` : ''}${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>` : ''}</section>`).join('')}</div></article><div class="attention-activity"><div class="activity-heading"><div><span class="lesson-label">TRY THE IDEA</span><h3>Ask one word what it needs.</h3><p>Select a query word. The lines show an educational example of where that word could look for context.</p></div><span class="activity-badge">ILLUSTRATIVE WEIGHTS</span></div><div class="query-picker">${demoTokens.map((token, index) => `<button class="query-button ${index === transformerState.query ? 'active' : ''}" data-query-token="${index}">${token}</button>`).join('')}</div><div class="transformer-lab-grid"><div class="visual-panel"><div class="panel-topline"><span class="panel-label">LIVE ATTENTION VIEW</span><span class="panel-note">Not learned model weights</span></div><canvas id="transformer-canvas" aria-label="Illustrative attention links for a selected query token"></canvas><div class="canvas-legend"><span><b class="dot q"></b>Query token</span><span><b class="dot k"></b>Context token</span><span><b class="dot state-dot-legend"></b>Stronger illustrative link</span></div></div><aside class="control-panel transformer-controls"><div class="control-block"><label for="transformer-heads"><span>Attention heads shown</span><strong>${transformerState.heads}</strong></label><input id="transformer-heads" type="range" min="1" max="8" value="${transformerState.heads}"><div class="range-labels"><span>1</span><span>8</span></div></div><div class="control-block"><label for="transformer-depth"><span>Stacked blocks</span><strong>${transformerState.depth}</strong></label><input id="transformer-depth" type="range" min="1" max="48" value="${transformerState.depth}"><div class="range-labels"><span>1</span><span>48</span></div></div><div class="architecture-select"><span class="control-caption">What happens in this view</span><div class="transformer-path-detail"><span class="lesson-label">WHAT THIS STAGE DOES</span><strong>${transformerPathDetails[transformerState.path][0]}</strong><p>${transformerPathDetails[transformerState.path][1]}</p><code>${transformerPathDetails[transformerState.path][2]}</code></div></div></aside></div><p class="activity-explanation"><strong>${demoTokens[transformerState.query]}</strong> is the query. Its attention weights are not computed from a trained model here; they are a small teaching example that makes the Q → K → V flow visible.</p></div><div class="takeaway-grid"><div class="takeaway-card"><span class="lesson-label">ONE-SENTENCE TAKEAWAY</span><strong>${lesson.takeaway}</strong></div><div class="check-card"><span class="lesson-label">MENTAL MODEL</span><p>Input representations move through repeated transformations. Attention communicates across positions; MLP layers transform each position; residuals carry the running signal forward.</p></div></div></div></div></section><section class="reference section-shell"><div><p class="eyebrow">Module bridge</p><h2>Now follow the memory.</h2><p>Once you understand how attention creates and uses Key and Value representations, continue to the KV-cache module to see why long context becomes a serving problem.</p></div><div class="equation-card"><span>NEXT MODULE</span><code>Transformer blocks<br>↓<br>attention history<br>↓<br>KV cache pressure</code><a class="primary-button" href="#top">Open KV cache <span>→</span></a></div></section></main><footer class="footer section-shell"><span>AI SYSTEMS LAB / MODULE 01</span><a class="text-button" href="#top">Go to KV cache →</a></footer>`;
  bindTransformerEvents();
  drawTransformerCanvas();
}

function bindTransformerEvents() {
  const sidebarHeading = document.querySelector<HTMLElement>('.transformer-section .sidebar-heading');
  if (sidebarHeading && !sidebarHeading.querySelector('[data-action="toggleEli5"]')) {
    const toggle = document.createElement('button');
    toggle.className = `eli5-toggle ${transformerState.eli5 ? 'active' : ''}`;
    toggle.dataset.action = 'toggleEli5';
    toggle.setAttribute('aria-pressed', String(transformerState.eli5));
    toggle.innerHTML = `ELI5 <span>${transformerState.eli5 ? 'ON' : 'OFF'}</span>`;
    sidebarHeading.appendChild(toggle);
  }
  const activeLesson = transformerLessons[transformerState.lesson];
  const copy = transformerState.eli5 ? transformerEli5[activeLesson.id] : activeLesson;
  const sectionIntro = document.querySelector<HTMLElement>('.transformer-section .section-intro');
  const readingIntro = document.querySelector<HTMLElement>('.transformer-section .reading-intro');
  const readingSections = document.querySelector<HTMLElement>('.transformer-section .reading-sections');
  const takeaway = document.querySelector<HTMLElement>('.transformer-section .takeaway-card strong');
  if (sectionIntro) sectionIntro.textContent = `${copy.question} ${copy.intro}`;
  if (readingIntro) readingIntro.textContent = copy.intro;
  if (readingSections) readingSections.innerHTML = copy.sections.map((section) => `<section class="reading-section"><h4>${section.heading}</h4><p>${section.body}</p>${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>` : ''}</section>`).join('');
  if (takeaway) takeaway.textContent = copy.takeaway;
  addResearchShelf();
  const pathList = document.querySelector<HTMLDivElement>('.transformer-path-list');
  if (pathList) pathList.innerHTML = transformerPath.map((path) => `<span>${path}</span>`).join('');
  let detailHost = document.querySelector<HTMLElement>('.transformer-path-detail');
  if (!detailHost && pathList?.parentElement) {
    detailHost = document.createElement('div');
    detailHost.className = 'transformer-path-detail';
    pathList.parentElement.appendChild(detailHost);
  }
  if (detailHost) {
    const [title, body, data] = transformerPathDetails[transformerState.path];
    detailHost.innerHTML = `<span class="lesson-label">WHAT THIS STAGE DOES</span><strong>${title}</strong><p>${body}</p><code>${data}</code>`;
  }
  document.querySelectorAll<HTMLButtonElement>('[data-transformer-lesson]').forEach((button) => button.addEventListener('click', () => { transformerState.lesson = Number(button.dataset.transformerLesson); renderTransformer(); }));
  document.querySelector<HTMLButtonElement>('[data-action="previousTransformer"]')?.addEventListener('click', () => changeTransformerLesson(-1));
  document.querySelector<HTMLButtonElement>('[data-action="nextTransformer"]')?.addEventListener('click', () => changeTransformerLesson(1));
  document.querySelector<HTMLButtonElement>('[data-action="startTransformer"]')?.addEventListener('click', () => document.querySelector('#transformer-lab')?.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth' }));
  document.querySelectorAll<HTMLButtonElement>('[data-query-token]').forEach((button) => button.addEventListener('click', () => { transformerState.query = Number(button.dataset.queryToken); renderTransformer(); }));
  document.querySelector<HTMLInputElement>('#transformer-heads')?.addEventListener('input', (event) => { transformerState.heads = Number((event.target as HTMLInputElement).value); renderTransformer(); });
  document.querySelector<HTMLInputElement>('#transformer-depth')?.addEventListener('input', (event) => { transformerState.depth = Number((event.target as HTMLInputElement).value); renderTransformer(); });
  document.querySelector<HTMLButtonElement>('[data-action="motion"]')?.addEventListener('click', () => { state.reducedMotion = !state.reducedMotion; renderTransformer(); });
  document.querySelector<HTMLButtonElement>('[data-action="toggleEli5"]')?.addEventListener('click', () => { transformerState.eli5 = !transformerState.eli5; renderTransformer(); });
  document.querySelectorAll<HTMLButtonElement>('[data-transformer-path]').forEach((button) => button.addEventListener('click', () => { transformerState.path = Number(button.dataset.transformerPath); renderTransformer(); }));
  document.querySelectorAll<HTMLElement>('.transformer-path-list span').forEach((stage, index) => {
    stage.classList.toggle('active', index === transformerState.path);
    stage.tabIndex = 0;
    stage.setAttribute('role', 'button');
    stage.setAttribute('aria-label', `View ${transformerPath[index]}`);
    stage.addEventListener('click', () => { transformerState.path = index; renderTransformer(); });
    stage.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); transformerState.path = index; renderTransformer(); } });
  });
}

function addResearchShelf() {
  const reference = document.querySelector<HTMLElement>('#transformer-top .reference');
  if (!reference || document.querySelector('#papers')) return;
  const paper = researchPapers[0];
  const section = document.createElement('section');
  section.id = 'papers';
  section.className = 'paper-section section-shell';
  section.innerHTML = `<div class="paper-heading"><div><p class="eyebrow">Research shelf</p><h2>Read the papers behind the ideas.</h2><p>Research papers introduce mechanisms; this shelf turns one paper at a time into a readable engineering story.</p></div><span class="paper-count">PAPER 01 / ${researchPapers.length}</span></div><article class="paper-card"><div class="paper-meta"><span>${paper.year}</span><span>${paper.citation}</span></div><h3>${paper.title}</h3><p class="paper-authors">${paper.authors}</p><p class="paper-question">${paper.question}</p><p class="paper-summary">${paper.summary}</p><div class="paper-ideas">${paper.ideas.map(([title, body], index) => `<div class="paper-idea"><span>0${index + 1}</span><strong>${title}</strong><p>${body}</p></div>`).join('')}</div><div class="paper-caveat"><span class="lesson-label">READ IT CAREFULLY</span><p>${paper.caveat}</p></div></article>`;
  reference.before(section);
}

function changeTransformerLesson(direction: number) {
  transformerState.lesson = Math.max(0, Math.min(transformerLessons.length - 1, transformerState.lesson + direction));
  renderTransformer();
  document.querySelector('#transformer-lab')?.scrollIntoView({ behavior: state.reducedMotion ? 'auto' : 'smooth', block: 'start' });
}

function attentionStrength(query: number, target: number): number {
  if (query === 4 && target === 1) return 0.9;
  if (query === 4 && target === 6) return 0.62;
  if (Math.abs(query - target) === 1) return 0.38;
  return 0.14;
}

function drawTransformerCanvas() {
  const canvas = document.querySelector<HTMLCanvasElement>('#transformer-canvas');
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, bounds.width * ratio); canvas.height = Math.max(1, bounds.height * ratio);
  const context = canvas.getContext('2d'); if (!context) return;
  context.scale(ratio, ratio); const width = bounds.width; const height = bounds.height;
  context.fillStyle = '#10252d'; context.fillRect(0, 0, width, height); context.strokeStyle = 'rgba(168, 220, 203, .10)';
  for (let x = 0; x < width; x += 36) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y < height; y += 36) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.font = '11px "DM Mono", monospace'; context.fillStyle = '#8ab6ad'; context.fillText(`${transformerPath[transformerState.path].toUpperCase()} / BLOCK ${transformerState.depth}`, 24, 28);
  const gap = Math.min(48, (width - 90) / transformerState.tokens); const start = (width - gap * transformerState.tokens) / 2; const tokenY = height * .28;
  context.font = '11px "DM Sans", sans-serif';
  for (let i = 0; i < transformerState.tokens; i++) { const x = start + i * gap; const isQuery = i === transformerState.query; context.fillStyle = isQuery ? '#b3c938' : '#6c5ac7'; context.beginPath(); context.roundRect(x, tokenY, 44, 28, 5); context.fill(); context.fillStyle = '#f1f6ee'; context.fillText(demoTokens[i] ?? `t${i + 1}`, x + 6, tokenY + 18); }
  context.strokeStyle = 'rgba(110, 210, 189, .48)'; context.lineWidth = 1;
  for (let i = 0; i < transformerState.tokens; i++) { if (i === transformerState.query) continue; const x1 = start + transformerState.query * gap + 22; const x2 = start + i * gap + 22; const strength = attentionStrength(transformerState.query, i); context.globalAlpha = .18 + strength * .68; context.lineWidth = 1 + strength * 3; context.beginPath(); context.moveTo(x1, tokenY + 30); context.lineTo(x2, tokenY + 90); context.stroke(); context.globalAlpha = 1; context.fillStyle = '#d8e775'; context.font = '10px "DM Mono", monospace'; context.fillText(`${Math.round(strength * 100)}%`, x2 - 11, tokenY + 105); }
  const blockY = height * .63; const blockWidth = Math.min(130, (width - 80) / 3); ['ATTENTION', 'MLP', 'RESIDUAL'].forEach((label, index) => { const x = 24 + index * (blockWidth + 18); const isSelected = (transformerState.path === 2 && index === 0) || (transformerState.path === 3 && index > 0) || (transformerState.path === 4 && index === 2); context.fillStyle = isSelected ? '#167d78' : '#55747b'; context.beginPath(); context.roundRect(x, blockY, blockWidth, 50, 7); context.fill(); context.fillStyle = '#f1f6ee'; context.font = '11px "DM Mono", monospace'; context.fillText(label, x + 14, blockY + 29); if (index < 2) { context.strokeStyle = '#d8e775'; context.beginPath(); context.moveTo(x + blockWidth, blockY + 25); context.lineTo(x + blockWidth + 18, blockY + 25); context.stroke(); } });
  context.fillStyle = '#8ab6ad'; context.font = '11px "DM Mono", monospace'; context.fillText('contextual representation → next-token scores', 24, height - 24);
}

function drawCanvas() {
  const canvas = document.querySelector<HTMLCanvasElement>('#lab-canvas');
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, bounds.width * ratio);
  canvas.height = Math.max(1, bounds.height * ratio);
  const context = canvas.getContext('2d');
  if (!context) return;
  context.scale(ratio, ratio);
  const width = bounds.width;
  const height = bounds.height;
  const m = metrics();
  context.clearRect(0, 0, width, height);
  const grid = context.createLinearGradient(0, 0, width, height);
  grid.addColorStop(0, '#132a35'); grid.addColorStop(1, '#0c1821'); context.fillStyle = grid; context.fillRect(0, 0, width, height);
  context.strokeStyle = 'rgba(168, 220, 203, .07)'; context.lineWidth = 1;
  for (let x = 0; x < width; x += 36) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y < height; y += 36) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.font = '11px "DM Mono", monospace'; context.fillStyle = '#86aaa7'; context.fillText(`${architectureLabels[state.architecture].toUpperCase()} / REPRESENTATION`, 24, 30);
  const baseY = height * .53;
  const visible = Math.min(28, Math.max(7, Math.round(Math.log10(state.tokens) * 8)));
  const startX = 32;
  const gap = Math.min(34, (width - 170) / visible);
  context.fillStyle = '#dbe8df'; context.font = '12px "DM Sans", sans-serif'; context.fillText(`${state.tokens.toLocaleString()} token positions`, 24, height - 25);
  for (let i = 0; i < visible; i++) {
    const x = startX + i * gap;
    const isActive = state.architecture !== 'attention' && i >= visible - Math.max(3, Math.round(visible * .28));
    context.fillStyle = isActive ? '#d8f26e' : '#55747b';
    context.beginPath(); context.roundRect(x, baseY - 10, 22, 20, 4); context.fill();
    context.fillStyle = isActive ? '#d8f26e' : '#7ca19b'; context.beginPath(); context.arc(x + 11, baseY - 36, 3, 0, Math.PI * 2); context.fill();
    context.fillStyle = isActive ? '#d8f26e' : '#8f7ee8'; context.beginPath(); context.arc(x + 11, baseY + 36, 3, 0, Math.PI * 2); context.fill();
    if (i < visible - 1) { context.strokeStyle = 'rgba(216, 242, 110, .15)'; context.beginPath(); context.moveTo(x + 22, baseY); context.lineTo(x + gap, baseY); context.stroke(); }
  }
  const boxX = Math.max(width - 250, startX + visible * gap + 24);
  const compact = state.architecture !== 'attention';
  const boxW = Math.min(210, width - boxX - 24);
  context.strokeStyle = compact ? '#d8f26e' : '#7ca19b'; context.lineWidth = 2; context.beginPath(); context.roundRect(boxX, baseY - 66, boxW, 132, 12); context.stroke();
  context.fillStyle = compact ? '#d8f26e' : '#b8ceca'; context.font = '12px "DM Mono", monospace'; context.fillText(compact ? 'COMPACT STATE' : 'KV CACHE', boxX + 18, baseY - 38);
  context.fillStyle = '#dbe8df'; context.font = '16px "DM Sans", sans-serif'; context.fillText(compact ? state.architecture === 'window' ? `${Math.min(state.tokens, state.window)} recent` : 'history → state' : `${state.tokens.toLocaleString()} K / V pairs`, boxX + 18, baseY + 4);
  context.fillStyle = '#86aaa7'; context.font = '11px "DM Mono", monospace'; context.fillText(compact ? `~${Math.round(m.bounded)}% bounded` : `× ${state.users} active users`, boxX + 18, baseY + 35);
  const pressureWidth = Math.min(width - 48, (width - 48) * m.pressure / 100);
  context.fillStyle = 'rgba(216,242,110,.12)'; context.fillRect(24, 52, width - 48, 5); context.fillStyle = m.pressure > 100 ? '#ff9770' : '#d8f26e'; context.fillRect(24, 52, pressureWidth, 5);
}

window.addEventListener('resize', () => { drawCanvas(); drawTransformerCanvas(); });
window.addEventListener('hashchange', render);
render();
