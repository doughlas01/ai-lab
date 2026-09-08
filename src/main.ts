import './style.css';
import { lessons } from './lessonData';
import { reading } from './lessonReading';
import { researchPapers, transformerLessons, tokenDetectiveData } from './transformerData';
import {
  attentionPaperChapters,
  attentionPaperEli5,
  upcomingPapers,
  telephoneVsLaserData,
  volumeDialDemo,
  eightGlassesData,
  rhythmClockData,
  translationRoomEvolution
} from './researchData';

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

const generatorState = {
  promptTokens: ['The', 'secret', 'key', 'is', 'ALPHA-99', '.', 'Please', 'confirm', 'key', ':'],
  generationTokens: ['ALPHA-99', '.', 'Access', 'granted', 'to', 'user', '.'],
  step: 0,
  useCache: true,
  isPlaying: false,
  timer: null as number | null,
  needleArch: 'attention' as 'attention' | 'window' | 'mamba'
};

const researchViewState = {
  mode: 'transformer' as 'rnn' | 'transformer',
  activeTrack: 'story' as 'story' | 'volume' | 'glasses' | 'clocks' | 'evolution',
  volumeMode: 'scaled' as 'scaled' | 'unscaled',
  activeGlassesId: 'glasses-1',
  activeClockWordIdx: 1,
  activeRoomId: 'room-3'
};

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

function getGeneratorMetrics() {
  const promptLen = generatorState.promptTokens.length;
  const currentGen = generatorState.generationTokens.slice(0, generatorState.step);
  const currentSeqLen = promptLen + generatorState.step;
  
  const stepCost = generatorState.useCache
    ? (generatorState.step === 0 ? promptLen : 1)
    : (generatorState.step === 0 ? promptLen : currentSeqLen);

  let withCacheTotal = promptLen + generatorState.step;
  let withoutCacheTotal = promptLen;
  for (let i = 1; i <= generatorState.step; i++) {
    withoutCacheTotal += (promptLen + i);
  }
  const wastedOps = withoutCacheTotal - withCacheTotal;

  const windowSize = 6;
  const windowStart = Math.max(0, currentSeqLen - windowSize);
  const needleIdx = 4;
  const isNeedleEvicted = generatorState.needleArch === 'window' && needleIdx < windowStart;

  let needleAnswer = 'ALPHA-99';
  let needleSuccess = true;
  let needleExplanation = 'Direct retrieval from VRAM memory slot 4.';

  if (generatorState.needleArch === 'window') {
    if (isNeedleEvicted) {
      needleAnswer = '[FORGOTTEN / HALLUCINATED: "ERR-00"]';
      needleSuccess = false;
      needleExplanation = `Position 4 fell outside the active ${windowSize}-token window! The model cannot attend to what was evicted.`;
    } else {
      needleAnswer = 'ALPHA-99';
      needleSuccess = true;
      needleExplanation = `Position 4 is still inside the active ${windowSize}-token window.`;
    }
  } else if (generatorState.needleArch === 'mamba') {
    needleAnswer = 'ALPHA-99';
    needleSuccess = true;
    needleExplanation = 'Input-dependent selective state recognized the key as important and preserved it in fixed-size state memory.';
  } else {
    needleAnswer = 'ALPHA-99';
    needleSuccess = true;
    needleExplanation = 'Full attention preserved all Key & Value activations in VRAM. Complete recall.';
  }

  return {
    promptLen,
    currentGen,
    currentSeqLen,
    stepCost,
    wastedOps,
    windowStart,
    windowSize,
    needleIdx,
    isNeedleEvicted,
    needleAnswer,
    needleSuccess,
    needleExplanation
  };
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
  const genM = getGeneratorMetrics();

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

        <!-- Step-by-Step Autoregressive Generator & VRAM Shelf Playground -->
        <section class="generator-section">
          <div class="generator-header">
            <div>
              <span class="lesson-label">STEP-BY-STEP GENERATION & VRAM SIMULATOR</span>
              <h3>Why do we need a KV Cache?</h3>
              <p>Watch autoregressive decoding in action. Toggle between saving activations vs. recomputing past words on every single step.</p>
            </div>
            <div class="cache-toggle-wrap">
              <span>CACHING MODE:</span>
              <button class="cache-mode-btn ${generatorState.useCache ? 'active' : ''}" data-cache-mode="cached">⚡ WITH KV CACHE</button>
              <button class="cache-mode-btn ${!generatorState.useCache ? 'active no-cache' : ''}" data-cache-mode="no-cache">⚠️ WITHOUT CACHE (RECOMPUTE)</button>
            </div>
          </div>

          <div class="generator-controls-bar">
            <div class="generator-btn-group">
              <button class="generator-btn primary" data-action="stepToken" ${generatorState.step >= generatorState.generationTokens.length ? 'disabled' : ''}>
                ▶ Step Next Token (${generatorState.step} / ${generatorState.generationTokens.length})
              </button>
              <button class="generator-btn" data-action="toggleAutoPlay">
                ${generatorState.isPlaying ? '⏸ Pause' : '⏯ Auto-play'}
              </button>
              <button class="generator-btn" data-action="resetGenerator">
                ↺ Reset Sequence
              </button>
            </div>
            <div class="cost-stat-chips">
              <span class="cost-chip">Sequence: <strong>${genM.currentSeqLen} tokens</strong></span>
              <span class="cost-chip">Current Step Compute: <strong>${genM.stepCost} token(s)</strong></span>
            </div>
          </div>

          <div class="cost-banner ${generatorState.useCache ? 'cached' : 'no-cache'}">
            <div>
              ${generatorState.useCache
                ? `<strong>⚡ O(1) EFFICIENT DECODE:</strong> Generating token <strong>"${generatorState.step === 0 ? 'prompt' : generatorState.generationTokens[generatorState.step - 1]}"</strong> only required computing <strong>1 new token</strong>. All previous ${genM.currentSeqLen - 1} Key & Value activations were loaded directly from the VRAM shelf!`
                : `<strong>⚠️ O(N) NAIVE RECOMPUTATION:</strong> Generating token <strong>"${generatorState.step === 0 ? 'prompt' : generatorState.generationTokens[generatorState.step - 1]}"</strong> forced the GPU to re-compute all <strong>${genM.currentSeqLen} tokens from scratch</strong>! Cumulative wasted compute: <strong>+${genM.wastedOps} FLOPs</strong>!`
              }
            </div>
          </div>

          <div class="sequence-viewer">
            <div class="sequence-label">
              <span>AUTOREGRESSIVE TOKEN STREAM</span>
              <span>${generatorState.step === 0 ? 'PREFILL STAGE (Prompt Input)' : 'DECODE STAGE (Token Generation)'}</span>
            </div>
            <div class="sequence-tokens">
              ${generatorState.promptTokens.map((tok, i) => `
                <span class="seq-token prompt ${i === 4 ? 'needle' : ''} ${!generatorState.useCache && generatorState.step > 0 ? 'recomputed' : ''}">
                  ${tok}${i === 4 ? ' 🔑' : ''}
                </span>
              `).join('')}
              ${genM.currentGen.map((tok, i) => `
                <span class="seq-token generated ${i === genM.currentGen.length - 1 ? 'newest' : ''} ${!generatorState.useCache ? 'recomputed' : ''}">
                  ${tok}
                </span>
              `).join('')}
              ${generatorState.step < generatorState.generationTokens.length ? '<span class="typing-cursor"></span>' : ''}
            </div>
          </div>

          <div class="shelf-container">
            <div class="shelf-header">
              <h4>VRAM Shelf: Stored Keys & Values (${genM.currentSeqLen} positions)</h4>
              <span class="shelf-capacity">${generatorState.useCache ? 'Allocated in GPU Memory' : 'Empty (No Cache Stored)'}</span>
            </div>
            <div class="shelf-slots">
              ${generatorState.useCache ? (
                generatorState.needleArch === 'mamba'
                  ? `
                    <div class="shelf-slot needle">
                      <div class="slot-token">ALPHA-99</div>
                      <div class="slot-type">STATE: KEY</div>
                    </div>
                    <div class="shelf-slot active">
                      <div class="slot-token">confirm</div>
                      <div class="slot-type">STATE: TASK</div>
                    </div>
                    <div class="shelf-slot active">
                      <div class="slot-token">${generatorState.step > 0 ? generatorState.generationTokens[generatorState.step - 1] : 'prompt'}</div>
                      <div class="slot-type">STATE: LATEST</div>
                    </div>
                  `
                  : Array.from({ length: genM.currentSeqLen }).map((_, idx) => {
                      const allTokens = [...generatorState.promptTokens, ...genM.currentGen];
                      const tok = allTokens[idx];
                      const isNeedle = idx === 4;
                      const isEvicted = generatorState.needleArch === 'window' && idx < genM.windowStart;
                      return `
                        <div class="shelf-slot ${isNeedle ? 'needle' : ''} ${isEvicted ? 'evicted' : 'active'}">
                          <div class="slot-token">${tok}</div>
                          <div class="slot-type">${isEvicted ? 'EVICTED' : 'K / V'}</div>
                        </div>
                      `;
                    }).join('')
              ) : `
                <div style="grid-column: 1 / -1; padding: 12px; text-align: center; color: var(--dim); font-family: 'DM Mono', monospace; font-size: 11px;">
                  No activations stored in VRAM. Every decoding step must re-run the entire sequence through all neural network layers from token 0.
                </div>
              `}
            </div>
          </div>

          <div class="needle-card">
            <div class="needle-top">
              <h4>Architecture Needle Test: "What was the secret key?"</h4>
              <div class="needle-arch-btns">
                <button class="needle-arch-btn ${generatorState.needleArch === 'attention' ? 'active' : ''}" data-needle-arch="attention">Full Attention</button>
                <button class="needle-arch-btn ${generatorState.needleArch === 'window' ? 'active' : ''}" data-needle-arch="window">Sliding Window (W=6)</button>
                <button class="needle-arch-btn ${generatorState.needleArch === 'mamba' ? 'active' : ''}" data-needle-arch="mamba">Mamba State</button>
              </div>
            </div>
            <div class="needle-result-box">
              <div class="needle-result-item">
                <span>MODEL RETRIEVAL OUTPUT</span>
                <strong class="${genM.needleSuccess ? 'success' : 'failure'}">${genM.needleAnswer}</strong>
                <p>${genM.needleExplanation}</p>
              </div>
              <div class="needle-result-item">
                <span>SYSTEM MEMORY STATUS</span>
                <strong>${generatorState.needleArch === 'attention' ? `${genM.currentSeqLen} slots in VRAM (O(N) growth)` : generatorState.needleArch === 'window' ? `${Math.min(genM.windowSize, genM.currentSeqLen)} slots in VRAM (Bounded)` : '3 state vectors (Fixed O(1) memory)'}</strong>
                <p>${generatorState.needleArch === 'attention' ? 'Perfect accuracy, but memory increases with every generated word.' : generatorState.needleArch === 'window' ? (genM.isNeedleEvicted ? 'Memory bounded, but old context was evicted and forgotten!' : 'Memory bounded; secret key is still within window.') : 'Fixed memory footprint with input-dependent selectivity preserving the key.'}</p>
              </div>
            </div>
          </div>
        </section>

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

  // Generator Playground Event Listeners
  document.querySelector<HTMLButtonElement>('[data-action="stepToken"]')?.addEventListener('click', () => {
    if (generatorState.step < generatorState.generationTokens.length) {
      generatorState.step++;
      render();
    }
  });

  document.querySelector<HTMLButtonElement>('[data-action="resetGenerator"]')?.addEventListener('click', () => {
    generatorState.step = 0;
    if (generatorState.timer) {
      clearInterval(generatorState.timer);
      generatorState.timer = null;
    }
    generatorState.isPlaying = false;
    render();
  });

  document.querySelector<HTMLButtonElement>('[data-action="toggleAutoPlay"]')?.addEventListener('click', () => {
    generatorState.isPlaying = !generatorState.isPlaying;
    if (generatorState.isPlaying) {
      if (generatorState.step >= generatorState.generationTokens.length) {
        generatorState.step = 0;
      }
      generatorState.timer = window.setInterval(() => {
        if (generatorState.step < generatorState.generationTokens.length) {
          generatorState.step++;
          render();
        } else {
          if (generatorState.timer) clearInterval(generatorState.timer);
          generatorState.timer = null;
          generatorState.isPlaying = false;
          render();
        }
      }, 950);
    } else {
      if (generatorState.timer) {
        clearInterval(generatorState.timer);
        generatorState.timer = null;
      }
    }
    render();
  });

  document.querySelectorAll<HTMLButtonElement>('[data-cache-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      generatorState.useCache = btn.dataset.cacheMode === 'cached';
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>('[data-needle-arch]').forEach((btn) => {
    btn.addEventListener('click', () => {
      generatorState.needleArch = btn.dataset.needleArch as 'attention' | 'window' | 'mamba';
      render();
    });
  });
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
  const paper = researchPapers[0];
  const telData = telephoneVsLaserData[researchViewState.mode];
  const telTokens = [
    { word: 'The', rnnState: 'Subject: Cat', status: 'active' },
    { word: 'cat', rnnState: 'Subject: Cat', status: 'active' },
    { word: 'sat', rnnState: 'Action: Sat', status: 'active' },
    { word: 'because', rnnState: 'Causal link', status: 'degraded' },
    { word: 'it', rnnState: 'Pronoun (blurry)', status: 'degraded' },
    { word: 'was', rnnState: 'Signal lost (???)', status: 'degraded' }
  ];

  const volScenario = volumeDialDemo.scenarios[researchViewState.volumeMode];
  const activeGlasses = eightGlassesData.find((g) => g.id === researchViewState.activeGlassesId) || eightGlassesData[0];
  const sentenceWords = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired'];
  const activeRoom = translationRoomEvolution.find((r) => r.id === researchViewState.activeRoomId) || translationRoomEvolution[2];

  // Analog Clock angles calculation (degrees)
  const clockAngles = [
    (researchViewState.activeClockWordIdx * 65) % 360,
    (researchViewState.activeClockWordIdx * 24) % 360,
    (researchViewState.activeClockWordIdx * 6) % 360
  ];

  root.innerHTML = `
    <header class="topbar"><a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a><div class="topbar-meta"><a href="#transformer">MODULE 01 / TRANSFORMER</a><span class="status-dot"></span><a href="#top">MODULE 02 / KV CACHE</a><a href="#research">RESEARCH PAPERS</a></div><button class="quiet-button" data-action="motion">${state.reducedMotion ? 'Motion off' : 'Reduce motion'}</button></header>
    <main id="research-top">
      <section class="research-hero section-shell">
        <div>
          <p class="eyebrow">Research paper 01 / Foundations track</p>
          <h1>Attention Is <em>All You Need.</em></h1>
          <p class="hero-lede">The breakthrough 2017 paper explained through simple, intuitive analogies. Learn why older models were stuck in a single-file game of telephone, how the Bilingual Newsroom replaced recurrence, and why dividing by √dₖ prevented one word from screaming.</p>
          <div class="paper-meta research-hero-meta"><span>${paper.year}</span><span>${paper.citation}</span><span>Vaswani et al.</span></div>
        </div>
        <div class="research-thesis">
          <span class="lesson-label">THE PAPER IN ONE ANALOGY</span>
          <strong>Knock down the hallway walls: let every word talk and listen to every other word simultaneously.</strong>
          <div class="research-thesis-flow"><span>50 people whispering</span><b>→</b><span>open room with laser pointers</span></div>
        </div>
      </section>

      <!-- Interactive Telephone vs Laser Pointer Simulator -->
      <section class="tel-laser-section section-shell">
        <div class="tel-laser-header">
          <div>
            <span class="lesson-label">THE FOUNDATIONAL ANALOGY</span>
            <h3>The Telephone Game vs. The Laser Pointer</h3>
            <p>Why did the 2017 Transformer replace RNNs? Switch paradigms below to see the hardware bottleneck and memory loss.</p>
          </div>
          <div class="tel-mode-switcher">
            <button class="tel-mode-btn ${researchViewState.mode === 'rnn' ? 'active' : ''}" data-tel-mode="rnn">1986–2016: The Telephone Game (RNN)</button>
            <button class="tel-mode-btn ${researchViewState.mode === 'transformer' ? 'active' : ''}" data-tel-mode="transformer">2017: The Laser Pointer (Transformer)</button>
          </div>
        </div>

        <div class="tel-stage">
          <div class="tel-stage-label">
            <span>SEQUENCE FLOW: "The cat sat because it was tired"</span>
            <span>${telData.era}</span>
          </div>
          <div class="tel-nodes-wrap">
            ${telTokens.map((item, idx) => `
              <div class="tel-node ${researchViewState.mode === 'rnn' ? item.status : (idx === 1 || idx === 4 ? 'laser-target' : 'active')}">
                <div class="tel-node-circle">${String(idx + 1).padStart(2, '0')}</div>
                <span class="tel-node-token">“${item.word}”</span>
                <span class="tel-node-whisper">${researchViewState.mode === 'rnn' ? item.rnnState : (idx === 4 ? 'Query: “it”' : idx === 1 ? 'Key: “cat” 🎯' : 'Parallel Q/K/V')}</span>
              </div>
              ${idx < telTokens.length - 1 ? `
                <div class="tel-connector-line ${researchViewState.mode === 'rnn' ? 'whisper-arrow' : 'laser-ray'}"></div>
              ` : ''}
            `).join('')}
          </div>
          <div class="tel-gpu-meter">
            <span>SUPERCOMPUTER HARDWARE BOTTLENECK:</span>
            <strong class="${researchViewState.mode === 'transformer' ? 'good' : 'bad'}">
              ${researchViewState.mode === 'transformer' ? '⚡ 10,000 / 10,000 GPU CORES SATURATED (All words look at once in parallel)' : '⚠️ 1 / 10,000 GPU CORES ACTIVE (Word 6 waits in line for Word 5 to whisper)'}
            </strong>
          </div>
        </div>

        <div class="tel-points-grid">
          ${telData.points.map((pt) => `
            <div class="tel-point-card">
              <span class="kicker">${pt.kicker}</span>
              <h4>${pt.title}</h4>
              <p>${pt.desc}</p>
            </div>
          `).join('')}
        </div>

        <div class="tel-catch-card">
          <span class="kicker">THE 2024 INFERENCE CATCH (CONNECTING THEORY TO REALITY)</span>
          <h4>Why the laser pointer created the KV Cache problem:</h4>
          <p>In training, laser pointers are computed all at once in parallel because the whole sentence is known upfront. But in live chat generation (inference), you must produce one token at a time. Storing every past laser target in memory for each user creates the massive KV Cache serving crisis you explored in Module 2!</p>
        </div>
      </section>

      <!-- 5-Track Interactive Analogy Navigation -->
      <nav class="research-track-nav section-shell" aria-label="Research Paper Tracks">
        <button class="research-track-btn ${researchViewState.activeTrack === 'story' ? 'active' : ''}" data-research-track="story">
          <span>TRACK 01</span>
          <strong>01. The Newsroom Story</strong>
        </button>
        <button class="research-track-btn ${researchViewState.activeTrack === 'volume' ? 'active' : ''}" data-research-track="volume">
          <span>TRACK 02</span>
          <strong>02. The Volume Dial (√dₖ)</strong>
        </button>
        <button class="research-track-btn ${researchViewState.activeTrack === 'glasses' ? 'active' : ''}" data-research-track="glasses">
          <span>TRACK 03</span>
          <strong>03. The 8 Pairs of Glasses</strong>
        </button>
        <button class="research-track-btn ${researchViewState.activeTrack === 'clocks' ? 'active' : ''}" data-research-track="clocks">
          <span>TRACK 04</span>
          <strong>04. The Orchestra Clocks</strong>
        </button>
        <button class="research-track-btn ${researchViewState.activeTrack === 'evolution' ? 'active' : ''}" data-research-track="evolution">
          <span>TRACK 05</span>
          <strong>05. From 2017 to ChatGPT</strong>
        </button>
      </nav>

      <!-- ACTIVE TRACK CONTENT -->
      ${researchViewState.activeTrack === 'story' ? `
        <!-- TRACK 01: Narrative Newsroom Chapters -->
        <section class="paper-reader section-shell">
          <aside class="paper-reader-nav">
            <div class="sidebar-heading">
              <span class="lesson-label">THE STORY MAP</span>
              <strong>8 Paper Chapters</strong>
            </div>
            <nav aria-label="Research paper chapters">
              ${attentionPaperChapters.map((item, index) => `
                <button class="paper-chapter ${index === state.chapter % attentionPaperChapters.length ? 'active' : ''}" data-research-chapter="${index}">
                  <span>${String(index + 1).padStart(2, '0')}</span>
                  <strong>${item.title}</strong>
                  <small>${item.label}</small>
                </button>
              `).join('')}
            </nav>
          </aside>
          <article class="paper-reading">
            <div class="paper-reading-header">
              <div>
                <span class="lesson-label">CHAPTER ${String((state.chapter % attentionPaperChapters.length) + 1).padStart(2, '0')} · ${chapter.label.toUpperCase()}</span>
                <h2>${chapter.title}</h2>
              </div>
              <div class="lesson-step-actions">
                <button class="step-button" data-action="previousResearch" ${state.chapter === 0 ? 'disabled' : ''}>← Previous</button>
                <span>${(state.chapter % attentionPaperChapters.length) + 1} / ${attentionPaperChapters.length}</span>
                <button class="step-button" data-action="nextResearch" ${state.chapter === attentionPaperChapters.length - 1 ? 'disabled' : ''}>Next →</button>
              </div>
            </div>

            <!-- Analogy Highlight Card -->
            <div style="background: rgba(110, 210, 189, 0.08); border-left: 3px solid var(--teal); padding: 14px 18px; margin: 20px 0 16px; border-radius: 0 4px 4px 0;">
              <span class="lesson-label">THE INTUITIVE ANALOGY</span>
              <p style="margin: 4px 0 0; font-size: 14px; color: var(--ink); line-height: 1.5;">${chapter.analogy}</p>
            </div>

            <p class="paper-reading-body">${state.eli5 ? attentionPaperEli5[state.chapter % attentionPaperEli5.length][1] : chapter.body}</p>

            <div class="paper-takeaway">
              <span class="lesson-label" style="color: #43540a;">KEY TAKEAWAY TO REMEMBER</span>
              <strong>${state.eli5 ? 'Think of it as giving every word a chance to look around for helpful clues.' : chapter.takeaway}</strong>
            </div>

            <div class="paper-activity">
              <span class="lesson-label">INTERACTIVE LAB TRACKS</span>
              <p>Want to see the physical metaphors in action? Explore the specialized analogy tracks above.</p>
              <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                <button class="primary-button" style="padding: 10px 14px; font-size: 12px;" data-research-track="volume">Try the Volume Dial 🎙️</button>
                <button class="primary-button" style="padding: 10px 14px; font-size: 12px; background: transparent; color: var(--teal); border-color: var(--teal);" data-research-track="glasses">Try the 8 Pairs of Glasses 👓</button>
              </div>
            </div>
          </article>
        </section>
      ` : ''}

      ${researchViewState.activeTrack === 'volume' ? `
        <!-- TRACK 02: The Microphone Volume Dial -->
        <section class="volume-dial-section section-shell">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Track 02 / The Volume Dial</p>
              <h2>The Microphone Volume Dial (Why Scaling by √dₖ is Required)</h2>
              <p class="section-intro" style="margin: 14px 0 0;">The Chorus vs. The Screaming Contest: How a simple volume dial saved AI from going deaf.</p>
            </div>
          </div>

          <!-- Analogy Story Card -->
          <div style="background: #09131a; border: 1px solid var(--line); border-left: 4px solid var(--lime); padding: 18px 22px; border-radius: 4px; margin: 20px 0 24px;">
            <span class="lesson-label">THE STORY OF THE SCREAMING CONTEST</span>
            <p style="font-size: 14px; color: var(--ink); line-height: 1.6; margin: 8px 0 0;">${volumeDialDemo.analogyStory}</p>
          </div>

          <!-- Volume Mode Toggle -->
          <div class="volume-toggle-bar">
            <button class="volume-mode-btn ${researchViewState.volumeMode === 'scaled' ? 'active scaled' : ''}" data-volume-mode="scaled">
              🟢 BALANCED VOLUME (Scaling ON: divide by √dₖ)
            </button>
            <button class="volume-mode-btn ${researchViewState.volumeMode === 'unscaled' ? 'active unscaled' : ''}" data-volume-mode="unscaled">
              🔴 VOLUME TOO LOUD (Scaling OFF: Raw Unscaled Shouting)
            </button>
          </div>

          <!-- Status Banner -->
          <div class="volume-status-banner ${researchViewState.volumeMode}">
            <strong style="display: block; font-family: 'Space Grotesk', sans-serif; font-size: 16px; margin-bottom: 4px;">${volScenario.status}</strong>
            <p style="margin: 0; font-size: 13px; line-height: 1.5;">${volScenario.explanation}</p>
          </div>

          <!-- Vocal Volume Bars Table -->
          <div class="volume-bars-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); border-bottom: 1px solid var(--line); padding-bottom: 10px;">
              <span>WORD (KEY CANDIDATE)</span>
              <span>HOW MUCH ATTENTION DOES "it" GIVE THIS WORD?</span>
            </div>

            <div class="volume-bars-table">
              ${volScenario.weights.map((item) => `
                <div class="volume-bar-row">
                  <span class="volume-word">“${item.word}”</span>
                  <span class="volume-role">${item.role}</span>
                  <div class="volume-track">
                    <div class="volume-fill ${researchViewState.volumeMode} ${item.isMain ? 'main-winner' : ''}" style="width: ${item.pct}%;"></div>
                  </div>
                  <span class="volume-pct">${item.pct}%</span>
                </div>
              `).join('')}
            </div>

            <!-- Gradient Health Box -->
            <div style="margin-top: 20px; padding: 14px 16px; background: rgba(0,0,0,0.3); border-radius: 4px; border: 1px solid var(--line);">
              <span class="lesson-label" style="color: ${researchViewState.volumeMode === 'scaled' ? 'var(--teal)' : 'var(--coral)'};">
                LEARNING HEALTH STATUS:
              </span>
              <p style="font-size: 13px; font-family: 'DM Mono', monospace; margin: 4px 0 0; color: var(--ink);">
                ${volScenario.gradientHealth}
              </p>
            </div>
          </div>
        </section>
      ` : ''}

      ${researchViewState.activeTrack === 'glasses' ? `
        <!-- TRACK 03: The 8 Pairs of Glasses -->
        <section class="glasses-section section-shell">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Track 03 / Multi-Head Attention</p>
              <h2>The 8 Pairs of Glasses (Multi-Head Attention)</h2>
              <p class="section-intro" style="margin: 14px 0 0;">You can't read a complex story with only one pair of eyes. Click each pair of glasses below to see what specialized clue that head inspects in the sentence.</p>
            </div>
          </div>

          <!-- Sentence Spotlight Bar -->
          <div class="spotlight-sentence-box">
            <div class="spotlight-sentence-label">
              <span>SENTENCE INSPECTED BY: <strong style="color: ${activeGlasses.color};">${activeGlasses.name.toUpperCase()}</strong></span>
              <span>Head 0${activeGlasses.headNumber} of 08</span>
            </div>

            <div class="spotlight-tokens-wrap">
              ${sentenceWords.map((word) => {
                const isFocused = activeGlasses.focusedWords.includes(word);
                return `
                  <span class="spotlight-token ${isFocused ? 'focused' : ''}" style="${isFocused ? `background: ${activeGlasses.color}; color: #09131a; border-color: ${activeGlasses.color};` : ''}">
                    ${word}
                    ${isFocused ? `<b style="position: absolute; top: -8px; right: -4px; width: 8px; height: 8px; border-radius: 50%; background: #ffffff;"></b>` : ''}
                  </span>
                `;
              }).join('')}
            </div>
          </div>

          <!-- 8 Glasses Selector Grid -->
          <div class="glasses-selector-grid">
            ${eightGlassesData.map((g) => `
              <button class="glasses-card-btn ${g.id === researchViewState.activeGlassesId ? 'active' : ''}" data-glasses-id="${g.id}">
                <span class="head-num">HEAD 0${g.headNumber} · ${g.badge}</span>
                <strong>${g.name}</strong>
              </button>
            `).join('')}
          </div>

          <!-- Active Glasses Detail Panel -->
          <div class="glasses-detail-box" style="border-left-color: ${activeGlasses.color};">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
              <h3 style="margin: 0; font-size: 20px; font-family: 'Space Grotesk', sans-serif; color: var(--ink);">
                Head 0${activeGlasses.headNumber}: ${activeGlasses.name}
              </h3>
              <span style="font: 11px 'DM Mono', monospace; color: ${activeGlasses.color};">${activeGlasses.badge}</span>
            </div>

            <p style="color: var(--ink); font-size: 15px; line-height: 1.5; margin: 8px 0 14px;">
              <strong>Question It Asks:</strong> "${activeGlasses.question}"
            </p>

            <div style="background: rgba(0,0,0,0.3); padding: 12px 14px; border-radius: 4px; margin-bottom: 14px;">
              <span class="lesson-label">WHAT THIS HEAD FOUND IN THE SENTENCE</span>
              <p style="font-size: 13px; color: var(--muted); line-height: 1.5; margin: 4px 0 0;">${activeGlasses.explanation}</p>
            </div>

            <div style="background: rgba(255,255,255,0.04); padding: 12px 14px; border-radius: 4px;">
              <span class="lesson-label">THE REAL-WORLD ANALOGY</span>
              <p style="font-size: 13px; color: var(--ink); line-height: 1.5; margin: 4px 0 0;">${activeGlasses.analogy}</p>
            </div>
          </div>
        </section>
      ` : ''}

      ${researchViewState.activeTrack === 'clocks' ? `
        <!-- TRACK 04: The Orchestra Clocks -->
        <section class="clocks-section section-shell">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Track 04 / Positional Encodings</p>
              <h2>The Orchestra Clocks (Positional Encodings Made Intuitive)</h2>
              <p class="section-intro" style="margin: 14px 0 0;">Without an order, words are just Scrabble tiles scattered on a table. See how musical rhythm clocks give every word a timestamp without counting.</p>
            </div>
          </div>

          <!-- Analogy Intro -->
          <div style="background: #09131a; border: 1px solid var(--line); border-left: 4px solid var(--violet); padding: 18px 22px; border-radius: 4px; margin: 20px 0 24px;">
            <span class="lesson-label">THE SCRABBLE TILE DILEMMA</span>
            <p style="font-size: 14px; color: var(--ink); line-height: 1.6; margin: 8px 0 0;">${rhythmClockData.story}</p>
          </div>

          <!-- Word Picker -->
          <div>
            <span class="lesson-label">CLICK ANY WORD TO INSPECT ITS RHYTHM CLOCK TIMESTAMPS:</span>
            <div class="clock-token-picker">
              ${sentenceWords.map((word, idx) => `
                <button class="clock-token-btn ${idx === researchViewState.activeClockWordIdx ? 'active' : ''}" data-clock-word="${idx}">
                  ${idx}: "${word}"
                </button>
              `).join('')}
            </div>
          </div>

          <!-- 3 Analog Clock Dials Grid -->
          <div class="clock-faces-grid">
            ${rhythmClockData.clockTypes.map((clock, i) => {
              const angleDeg = clockAngles[i];
              const angleRad = (angleDeg * Math.PI) / 180;
              const handX = 50 + 32 * Math.cos(angleRad - Math.PI / 2);
              const handY = 50 + 32 * Math.sin(angleRad - Math.PI / 2);
              return `
                <div class="clock-face-card">
                  <!-- Analog Clock SVG -->
                  <svg class="clock-dial-svg" viewBox="0 0 100 100" width="90" height="90" aria-label="Analog clock dial">
                    <circle cx="50" cy="50" r="44" fill="#0c1822" stroke="var(--line)" stroke-width="2"/>
                    <line x1="50" y1="10" x2="50" y2="16" stroke="var(--dim)" stroke-width="2"/>
                    <line x1="90" y1="50" x2="84" y2="50" stroke="var(--dim)" stroke-width="2"/>
                    <line x1="50" y1="90" x2="50" y2="84" stroke="var(--dim)" stroke-width="2"/>
                    <line x1="10" y1="50" x2="16" y2="50" stroke="var(--dim)" stroke-width="2"/>
                    <line x1="50" y1="50" x2="${handX}" y2="${handY}" stroke="${clock.color}" stroke-width="3.5" stroke-linecap="round"/>
                    <circle cx="50" cy="50" r="4" fill="${clock.color}"/>
                  </svg>
                  <h4>${clock.name}</h4>
                  <span class="clock-role">${clock.role} · ${clock.tempo}</span>
                  <p>${clock.desc}</p>
                  <div style="margin-top: 10px; font-family: 'DM Mono', monospace; font-size: 11px; color: ${clock.color};">
                    Word ${researchViewState.activeClockWordIdx} angle: ${angleDeg}°
                  </div>
                </div>
              `;
            }).join('')}
          </div>

          <!-- The Magic Insight Card -->
          <div style="background: rgba(154, 137, 255, 0.08); border: 1px solid rgba(154, 137, 255, 0.2); border-left: 3px solid var(--violet); padding: 18px 22px; border-radius: 4px;">
            <span class="lesson-label">THE MAGIC OF MUSICAL CLOCKS</span>
            <h4 style="font: 500 17px 'Space Grotesk', sans-serif; color: var(--ink); margin: 6px 0 8px;">
              ${rhythmClockData.magicInsight.heading}
            </h4>
            <p style="font-size: 13px; line-height: 1.6; color: var(--ink); margin: 0;">
              ${rhythmClockData.magicInsight.body}
            </p>
          </div>
        </section>
      ` : ''}

      ${researchViewState.activeTrack === 'evolution' ? `
        <!-- TRACK 05: From Newsroom to ChatGPT -->
        <section class="evolution-section section-shell">
          <div class="section-heading">
            <div>
              <p class="eyebrow">Track 05 / 2017 to ChatGPT</p>
              <h2>From the Bilingual Newsroom to Modern ChatGPT</h2>
              <p class="section-intro" style="margin: 14px 0 0;">How a 2017 translation architecture evolved into modern AI—and why it created the KV Cache crisis.</p>
            </div>
          </div>

          <!-- Era Cards -->
          <div class="newsroom-cards-grid">
            ${translationRoomEvolution.map((room) => `
              <div class="newsroom-card ${room.id === researchViewState.activeRoomId ? 'active' : ''}" data-room-id="${room.id}">
                <span class="era">${room.era}</span>
                <h4>${room.role}</h4>
                <p class="analogy-quote">“${room.analogy}”</p>
                <span class="models">${room.models}</span>
              </div>
            `).join('')}
          </div>

          <!-- Active Era Details -->
          <div style="background: #09131a; border: 1px solid var(--line); border-left: 4px solid var(--coral); padding: 22px 24px; border-radius: 4px; margin-bottom: 24px;">
            <span class="lesson-label">ACTIVE PARADIGM DEEP DIVE</span>
            <h3 style="margin: 6px 0 10px; font-size: 20px; font-family: 'Space Grotesk', sans-serif; color: var(--ink);">
              ${activeRoom.role} (${activeRoom.era})
            </h3>
            <p style="color: var(--ink); font-size: 14px; line-height: 1.6; margin: 0 0 16px;">
              ${activeRoom.howItWorks}
            </p>

            <div style="background: rgba(216,242,110,0.07); border: 1px solid rgba(216,242,110,0.2); padding: 14px 16px; border-radius: 4px;">
              <span class="lesson-label" style="color: var(--lime);">THE KV CACHE MEMORY REALITY</span>
              <p style="font-size: 13px; color: var(--ink); line-height: 1.5; margin: 4px 0 0;">
                ${activeRoom.kvStatus}
              </p>
            </div>
          </div>

          <!-- Bridge Callout to Module 02 -->
          <div style="background: #edf3d5; border-left: 4px solid #aabd31; padding: 20px 22px; border-radius: 4px;">
            <span style="font: 10px 'DM Mono', monospace; color: #667c13; letter-spacing: .08em; text-transform: uppercase; font-weight: 600;">
              THE FULL CIRCLE: FROM 2017 TRANSLATION TO 2024 KV CACHE SERVING
            </span>
            <h4 style="font: 500 18px 'Space Grotesk', sans-serif; color: #1a3014; margin: 6px 0 8px;">
              Why the Solo Writer Created the KV Cache Crisis
            </h4>
            <p style="font-size: 13px; line-height: 1.55; color: #43540a; margin: 0;">
              In 2017, translation used two separate desks (English and German). Modern AI (GPT-4, Claude, LLaMA) realized a single writer trained to guess the next word could do everything. But because the solo writer writes one word at a time, they must keep every past word’s notes on an expensive memory shelf. In long chats, this shelf overflows GPU memory—the exact bottleneck you explore in Module 02!
            </p>
            <div style="margin-top: 14px;">
              <a href="#top" style="color: #2b570e; font-weight: 700; font-size: 13px; text-decoration: none;">
                Jump to Module 02 (KV Cache Compression Lab) →
              </a>
            </div>
          </div>
        </section>
      ` : ''}

      <section class="upcoming-section section-shell">
        <div class="paper-heading">
          <div>
            <p class="eyebrow">Research shelf</p>
            <h2>More foundational papers, coming soon.</h2>
            <p>Each paper receives the same friendly treatment: the core problem, the big idea, vivid physical analogies, and systems engineering consequences.</p>
          </div>
          <span class="paper-count">${upcomingPapers.length} IN QUEUE</span>
        </div>
        <div class="upcoming-grid">
          ${upcomingPapers.map(([title, description], index) => `
            <article class="upcoming-card">
              <span>COMING SOON · 0${index + 2}</span>
              <h3>${title}</h3>
              <p>${description}</p>
            </article>
          `).join('')}
        </div>
      </section>
    </main>
    <footer class="footer section-shell">
      <span>AI SYSTEMS LAB / RESEARCH PAPERS</span>
      <a class="text-button" href="#transformer">Back to Transformer →</a>
    </footer>
  `;
  bindResearchEvents();
}

function bindResearchEvents() {
  // ELI5 Toggle
  const sidebarHeading = document.querySelector<HTMLElement>('.paper-reader-nav .sidebar-heading');
  if (sidebarHeading && !sidebarHeading.querySelector('[data-action="toggleEli5"]')) {
    const toggle = document.createElement('button');
    toggle.className = `eli5-toggle ${state.eli5 ? 'active' : ''}`;
    toggle.dataset.action = 'toggleEli5';
    toggle.setAttribute('aria-pressed', String(state.eli5));
    toggle.innerHTML = `ELI5 <span>${state.eli5 ? 'ON' : 'OFF'}</span>`;
    sidebarHeading.appendChild(toggle);
  }

  // Telephone / Laser Mode Switcher
  document.querySelectorAll<HTMLButtonElement>('[data-tel-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      researchViewState.mode = btn.dataset.telMode as 'rnn' | 'transformer';
      renderResearch();
    });
  });

  // Track Navigation Bar
  document.querySelectorAll<HTMLButtonElement>('[data-research-track]').forEach((btn) => {
    btn.addEventListener('click', () => {
      researchViewState.activeTrack = btn.dataset.researchTrack as any;
      renderResearch();
    });
  });

  // Track 01: Narrative Chapters
  document.querySelectorAll<HTMLButtonElement>('[data-research-chapter]').forEach((button) => {
    button.addEventListener('click', () => {
      state.chapter = Number(button.dataset.researchChapter);
      renderResearch();
    });
  });
  document.querySelector<HTMLButtonElement>('[data-action="previousResearch"]')?.addEventListener('click', () => {
    state.chapter = Math.max(0, state.chapter - 1);
    renderResearch();
  });
  document.querySelector<HTMLButtonElement>('[data-action="nextResearch"]')?.addEventListener('click', () => {
    state.chapter = Math.min(attentionPaperChapters.length - 1, state.chapter + 1);
    renderResearch();
  });
  document.querySelector<HTMLButtonElement>('[data-action="toggleEli5"]')?.addEventListener('click', () => {
    state.eli5 = !state.eli5;
    renderResearch();
  });

  // Track 02: Volume Mode
  document.querySelectorAll<HTMLButtonElement>('[data-volume-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      researchViewState.volumeMode = btn.dataset.volumeMode as 'scaled' | 'unscaled';
      renderResearch();
    });
  });

  // Track 03: 8 Pairs of Glasses
  document.querySelectorAll<HTMLButtonElement>('[data-glasses-id]').forEach((btn) => {
    btn.addEventListener('click', () => {
      researchViewState.activeGlassesId = btn.dataset.glassesId!;
      renderResearch();
    });
  });

  // Track 04: Orchestra Clocks Word Picker
  document.querySelectorAll<HTMLButtonElement>('[data-clock-word]').forEach((btn) => {
    btn.addEventListener('click', () => {
      researchViewState.activeClockWordIdx = Number(btn.dataset.clockWord);
      renderResearch();
    });
  });

  // Track 05: Evolution Rooms
  document.querySelectorAll<HTMLElement>('[data-room-id]').forEach((card) => {
    card.addEventListener('click', () => {
      researchViewState.activeRoomId = card.dataset.roomId!;
      renderResearch();
    });
  });

  document.querySelector<HTMLButtonElement>('[data-action="motion"]')?.addEventListener('click', () => {
    state.reducedMotion = !state.reducedMotion;
    renderResearch();
  });
}


function renderTransformer() {
  const lesson = transformerLessons[transformerState.lesson];
  const copy = transformerState.eli5 ? transformerEli5[lesson.id] : lesson;
  const paper = researchPapers[0];
  const detective = tokenDetectiveData[transformerState.query] ?? tokenDetectiveData[0];

  root.innerHTML = `
    <header class="topbar"><a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a><div class="topbar-meta"><a href="#transformer">MODULE 01 / TRANSFORMER</a><span class="status-dot"></span><a href="#top">MODULE 02 / KV CACHE</a><a href="#research" target="_blank" rel="noreferrer">RESEARCH PAPERS ↗</a></div><button class="quiet-button" data-action="motion">${state.reducedMotion ? 'Motion off' : 'Reduce motion'}</button></header>
    <main id="transformer-top"><section class="hero section-shell transformer-hero"><div class="hero-copy"><p class="eyebrow">Module 01 / Foundations</p><h1>How a Transformer <em>thinks in layers.</em></h1><p class="hero-lede">Follow a sequence from token IDs to contextual representations. See where attention connects positions, where the MLP transforms features, and why many blocks are stacked together.</p><div class="hero-actions"><button class="primary-button" data-action="startTransformer">Start the walkthrough <span>↗</span></button><a class="text-button" href="#top">Explore KV cache <span>→</span></a></div></div><div class="transformer-hero-diagram"><div class="diagram-caption">A SEQUENCE BECOMES A REPRESENTATION</div><div class="hero-token-row"><span>the</span><span>model</span><span>reads</span><span>context</span></div><div class="hero-arrow">↓</div><div class="hero-layer-row"><b>ATTENTION</b><b>MLP</b><b>ATTENTION</b></div><div class="hero-arrow">↓</div><div class="hero-output">contextual prediction</div></div></section>
    <section class="lab-section section-shell transformer-section" id="transformer-lab"><div class="lesson-layout"><aside class="lesson-sidebar"><div class="sidebar-heading"><span class="lesson-label">MODULE MAP</span><strong>Transformer path</strong></div><nav aria-label="Transformer lesson navigation">${transformerLessons.map((item, index) => `<button class="lesson-nav-item ${index === transformerState.lesson ? 'active' : ''}" data-transformer-lesson="${index}"><span>${item.kicker}</span><strong>${item.title}</strong><small>${item.question}</small></button>`).join('')}</nav></aside><div class="lesson-content"><div class="section-heading"><div><p class="eyebrow">${lesson.kicker}</p><h2>${lesson.title}</h2></div><div class="lesson-step-actions"><button class="step-button" data-action="previousTransformer" ${transformerState.lesson === 0 ? 'disabled' : ''}>← Previous</button><span>${transformerState.lesson + 1} / ${transformerLessons.length}</span><button class="step-button" data-action="nextTransformer" ${transformerState.lesson === transformerLessons.length - 1 ? 'disabled' : ''}>Next →</button></div></div><p class="section-intro">${lesson.question} ${lesson.intro}</p><article class="reading-panel"><div class="reading-header"><div><span class="lesson-label">READ THE CONCEPT</span><h3>${lesson.title}</h3></div><span class="reading-index">${String(transformerState.lesson + 1).padStart(2, '0')} / ${String(transformerLessons.length).padStart(2, '0')}</span></div><p class="reading-intro">${lesson.intro}</p><div class="reading-sections">${lesson.sections.map((section) => `<section class="reading-section"><h4>${section.heading}</h4><p>${section.body}</p>${section.formula ? `<code>${section.formula}</code>` : ''}${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>` : ''}</section>`).join('')}</div></article><div class="attention-activity"><div class="activity-heading"><div><span class="lesson-label">TRY THE IDEA</span><h3>Ask one word what it needs.</h3><p>Select a query word. The lines show an educational example of where that word could look for context.</p></div><span class="activity-badge">ILLUSTRATIVE WEIGHTS</span></div><div class="query-picker">${demoTokens.map((token, index) => `<button class="query-button ${index === transformerState.query ? 'active' : ''}" data-query-token="${index}">${token}</button>`).join('')}</div><div class="transformer-lab-grid"><div class="visual-panel"><div class="panel-topline"><span class="panel-label">LIVE ATTENTION VIEW</span><span class="panel-note">Illustrative attention weights</span></div><canvas id="transformer-canvas" aria-label="Illustrative attention links for a selected query token"></canvas><div class="canvas-legend"><span><b class="dot q"></b>Query token</span><span><b class="dot k"></b>Context token</span><span><b class="dot state-dot-legend"></b>Stronger illustrative link</span></div></div><aside class="control-panel transformer-controls"><div class="control-block"><label for="transformer-heads"><span>Attention heads shown</span><strong>${transformerState.heads}</strong></label><input id="transformer-heads" type="range" min="1" max="8" value="${transformerState.heads}"><div class="range-labels"><span>1</span><span>8</span></div></div><div class="control-block"><label for="transformer-depth"><span>Stacked blocks</span><strong>${transformerState.depth}</strong></label><input id="transformer-depth" type="range" min="1" max="48" value="${transformerState.depth}"><div class="range-labels"><span>1</span><span>48</span></div></div><div class="architecture-select"><span class="control-caption">What happens in this view</span><div class="transformer-path-detail"><span class="lesson-label">WHAT THIS STAGE DOES</span><strong>${transformerPathDetails[transformerState.path][0]}</strong><p>${transformerPathDetails[transformerState.path][1]}</p><code>${transformerPathDetails[transformerState.path][2]}</code></div></div></aside></div><p class="activity-explanation"><strong>"${demoTokens[transformerState.query]}"</strong> is the active query token. Attention weights below show how Query-Key matching pulls contextual meaning from other tokens.</p>

        <!-- Interactive Q-K-V Detective Board -->
        <div class="detective-board">
          <div class="detective-header">
            <div>
              <span class="lesson-label">INTERACTIVE Q-K-V DETECTIVE BOARD</span>
              <h4 style="margin:4px 0 0;font-size:19px;font-family:'Space Grotesk',sans-serif;color:var(--ink);">How attention connects words in practice</h4>
            </div>
            <div class="detective-token-badge">
              ACTIVE QUERY: <strong>"${demoTokens[transformerState.query]}"</strong> (Token ${transformerState.query + 1} of ${demoTokens.length})
            </div>
          </div>

          <div class="detective-query-box">
            <span class="query-label">🔎 1. THE QUERY (WHAT THIS WORD IS ASKING FOR)</span>
            <p class="query-text">${detective.queryQuestion}</p>
          </div>

          <div class="detective-keys-heading">
            <span class="lesson-label">🏷️ 2. MATCHING KEYS (CANDIDATE LABELS ACROSS THE SENTENCE)</span>
            <span style="font-size:11px;font-family:'DM Mono',monospace;color:var(--muted)">Click any word above to inspect its query</span>
          </div>

          <div class="detective-keys-grid">
            ${detective.keys.map((k) => `
              <div class="detective-key-card ${k.weight >= 0.7 ? 'high-match' : ''} ${k.targetIndex === transformerState.query ? 'query-self' : ''}">
                <div class="key-card-top">
                  <span class="key-card-word">“${demoTokens[k.targetIndex]}”</span>
                  <span class="key-card-score ${k.weight >= 0.7 ? 'high' : ''}">${Math.round(k.weight * 100)}% MATCH</span>
                </div>
                <div class="key-card-label">${k.keyLabel}</div>
                <div class="match-bar-wrap">
                  <div class="match-bar" style="width: ${Math.round(k.weight * 100)}%; background: ${k.weight >= 0.7 ? 'var(--teal)' : k.targetIndex === transformerState.query ? 'var(--violet)' : 'var(--dim)'}"></div>
                </div>
                <p class="key-card-reason">${k.reason}</p>
              </div>
            `).join('')}
          </div>

          <div class="detective-bottom-grid">
            <div class="detective-box">
              <h4>📦 3. VALUE RETRIEVED (THE MESSAGE EXTRACTED)</h4>
              <p>${detective.valuePayload}</p>
            </div>
            <div class="detective-box">
              <h4>🧬 4. CONTEXTUAL SYNTHESIS (UPDATED REPRESENTATION)</h4>
              <p>${detective.synthesis}</p>
            </div>
          </div>
        </div>

      </div><div class="takeaway-grid"><div class="takeaway-card"><span class="lesson-label">ONE-SENTENCE TAKEAWAY</span><strong>${lesson.takeaway}</strong></div><div class="check-card"><span class="lesson-label">MENTAL MODEL</span><p>Input representations move through repeated transformations. Attention communicates across positions; MLP layers transform each position; residuals carry the running signal forward.</p></div></div></div></div></section><section class="reference section-shell"><div><p class="eyebrow">Module bridge</p><h2>Now follow the memory.</h2><p>Once you understand how attention creates and uses Key and Value representations, continue to the KV-cache module to see why long context becomes a serving problem.</p></div><div class="equation-card"><span>NEXT MODULE</span><code>Transformer blocks<br>↓<br>attention history<br>↓<br>KV cache pressure</code><a class="primary-button" href="#top">Open KV cache <span>→</span></a></div></section></main><footer class="footer section-shell"><span>AI SYSTEMS LAB / MODULE 01</span><a class="text-button" href="#top">Go to KV cache →</a></footer>`;
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
  const detective = tokenDetectiveData[query];
  if (!detective) return 0.15;
  const match = detective.keys.find((k) => k.targetIndex === target);
  return match ? match.weight : 0.15;
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

window.addEventListener('resize', () => {
  drawCanvas();
  drawTransformerCanvas();
});
window.addEventListener('hashchange', render);
render();

