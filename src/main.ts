import './style.css';
import { lessons } from './lessonData';
import { reading } from './lessonReading';

type Architecture = 'attention' | 'window' | 'linear' | 'ssm' | 'mamba' | 'hybrid';

type State = {
  tokens: number;
  users: number;
  window: number;
  architecture: Architecture;
  reducedMotion: boolean;
  chapter: number;
};

const architectureLabels: Record<Architecture, string> = {
  attention: 'Standard attention',
  window: 'Sliding window',
  linear: 'Linear attention',
  ssm: 'State-space model',
  mamba: 'Mamba-style selectivity',
  hybrid: 'Hybrid stack'
};

const state: State = { tokens: 1280, users: 12, window: 128, architecture: 'attention', reducedMotion: false, chapter: 0 };
const model = { layers: 32, kvHeads: 8, headDim: 128, bytes: 2, gpu: 24, weights: 10 };
const root = document.querySelector<HTMLDivElement>('#app')!;

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
  const lesson = lessons[state.chapter];
  const lessonReading = reading[lesson.id];
  const m = metrics();
  root.innerHTML = `
    <div id="hover-card" class="hover-card" role="status" aria-live="polite"><span class="hover-card-kicker">CONTEXT NOTE</span><strong data-card-title></strong><p data-card-body></p></div>
    <header class="topbar">
      <a class="brand" href="#top" aria-label="AI Systems Lab home"><span class="brand-mark">AI</span><span>Systems Lab</span></a>
      <div class="topbar-meta"><span>MODULE 02</span><span class="status-dot"></span><span>KV CACHE / TOKEN HISTORY</span></div>
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
        <p class="section-intro">${lesson.question} ${lesson.explanation}</p>
        <article class="reading-panel"><div class="reading-header"><div><span class="lesson-label">READ THE CONCEPT</span><h3>${lesson.title}</h3></div><span class="reading-index">${String(state.chapter + 1).padStart(2, '0')} / ${String(lessons.length).padStart(2, '0')}</span></div><p class="reading-intro">${lessonReading.intro}</p><div class="reading-sections">${lessonReading.sections.map((section) => `<section class="reading-section"><h4>${section.heading}</h4><p>${section.body}</p>${section.formula ? `<code>${section.formula}</code>` : ''}${section.bullets ? `<ul>${section.bullets.map((bullet) => `<li>${bullet}</li>`).join('')}</ul>` : ''}</section>`).join('')}</div></article>
        <div class="lesson-strip"><div class="lesson-column"><span class="lesson-label">INTUITION</span><p>${lesson.analogy}</p></div><div class="lesson-column"><span class="lesson-label">WHAT TO WATCH</span><p>${lesson.visual}</p></div><div class="lesson-column"><span class="lesson-label">TRADE-OFF</span><p>${lesson.tradeoff}</p></div></div>
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
        <div class="takeaway-grid"><div class="takeaway-card"><span class="lesson-label">ONE-SENTENCE TAKEAWAY</span><strong>${lesson.takeaway}</strong></div><details class="check-card"><summary>Check your understanding</summary><p>${lesson.check.prompt}</p><span>${lesson.check.answer}</span></details></div>
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

window.addEventListener('resize', drawCanvas);
render();
