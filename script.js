// ════════════════════════════════════════════════════════════════
// Language Model Concepts — Probability of Sentences, Smoothing,
// Perplexity, Markov Models. Same design system + same corpus as
// the N-grams page, for continuity.
// ════════════════════════════════════════════════════════════════

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ─── DATA (shared corpus, same as the N-grams page) ────────────────────────
const CORPUS_TEXT = "language models predict the next word in a sentence . a good language model learns patterns from large amounts of text . n gram models count how often word sequences occur . bigram models look at pairs of words while trigram models look at three words at a time . the more context a model uses the better its predictions can be . however more context also means more data is needed to avoid sparsity . modern language models use neural networks instead of simple counts . but the core idea of predicting the next word remains the same .";

const IN_CORPUS_SAMPLES = [
  "language models predict the next word",
  "the more context a model uses",
  "bigram models look at pairs of words",
];
const OOV_SAMPLES = [
  "the cat sat on the mat",
  "she sells seashells by the seashore",
];

const N_META = {
  1: { label:"Unigram" },
  2: { label:"Bigram" },
  3: { label:"Trigram" },
};

const PAGE_HERO = {
  pill:"Foundations",
  sub:"Language Modeling",
  title:"Probability, Smoothing, Perplexity & Markov Models",
  desc:"How a language model scores a sentence, why it needs to hedge its bets, how we grade it, and the simple idea underneath all of it. One continuous walkthrough, using the same tiny corpus throughout.",
};

// ─── HELPERS ─────────────────────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s'-]/g, " ").split(/\s+/).filter(Boolean);
}

function getNgrams(tokens, n) {
  const grams = [];
  for (let i = 0; i <= tokens.length - n; i++) {
    grams.push({ start: i, words: tokens.slice(i, i + n) });
  }
  return grams;
}

function buildModel(tokens, n) {
  const model = {};
  for (let i = 0; i <= tokens.length - n; i++) {
    const context = n === 1 ? "" : tokens.slice(i, i + n - 1).join(" ");
    const nextWord = tokens[i + n - 1];
    if (!model[context]) model[context] = {};
    model[context][nextWord] = (model[context][nextWord] || 0) + 1;
  }
  return model;
}

function contextTotal(model, context) {
  const counts = model[context];
  if (!counts) return 0;
  return Object.values(counts).reduce((a, b) => a + b, 0);
}

function wordProb(model, context, word, smoothK, V) {
  const counts = model[context] || {};
  const c = counts[word] || 0;
  const total = contextTotal(model, context);
  if (smoothK > 0) {
    return (c + smoothK) / (total + smoothK * V);
  }
  return total > 0 ? c / total : 0;
}

// Computes P(sentence) via the chain rule, approximated with n-grams.
// Each step's context is the previous N-1 words of the *n-gram window*
// (a common simplification: the first N-1 words of the sentence are
// consumed as context rather than scored individually).
function sentenceProbability(sentenceTokens, n, model, smoothK, V) {
  const grams = getNgrams(sentenceTokens, n);
  const steps = grams.map(g => {
    const context = n === 1 ? "" : g.words.slice(0, n - 1).join(" ");
    const word = g.words[g.words.length - 1];
    const counts = model[context] || {};
    const count = counts[word] || 0;
    const ctxTotal = contextTotal(model, context);
    const p = wordProb(model, context, word, smoothK, V);
    return { context, word, count, ctxTotal, p };
  });
  const total = steps.reduce((acc, s) => acc * s.p, 1);
  return { steps, total };
}

function perplexity(total, length) {
  if (total <= 0 || length === 0) return Infinity;
  return Math.pow(total, -1 / length);
}

function fmtProb(p) {
  if (p === 0) return "0";
  if (p >= 0.001) return p.toFixed(4);
  return p.toExponential(2);
}

// ─── STATE ───────────────────────────────────────────────────────────────
const CORPUS_TOKENS = tokenize(CORPUS_TEXT);
const VOCAB_SIZE = new Set(CORPUS_TOKENS).size;
const MODELS = { 1: buildModel(CORPUS_TOKENS, 1), 2: buildModel(CORPUS_TOKENS, 2), 3: buildModel(CORPUS_TOKENS, 3) };

const STATE = {
  prob: { n: 2, sentence: IN_CORPUS_SAMPLES[0] },
  smooth: { n: 2, sentence: OOV_SAMPLES[0], k: 1 },
  perp: { sentence: IN_CORPUS_SAMPLES[0] },
  markov: { sequence: [], started: false },
};

// ─── APP SHELL (single continuous page, no sidebar) ────────────────────────
function initApp() {
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="sv-shell">
      <div class="sv-hero">
        <div class="sv-hero-pillrow">
          <span class="sv-hero-pill">${escapeHtml(PAGE_HERO.pill)}</span>
          <span class="sv-hero-pillsub">${escapeHtml(PAGE_HERO.sub)}</span>
        </div>
        <h1>${escapeHtml(PAGE_HERO.title)}</h1>
        <p>${escapeHtml(PAGE_HERO.desc)}</p>
      </div>
      <div class="sv-maininner" id="mainInner"></div>
    </div>
  `;
  document.getElementById("mainInner").innerHTML = `
    <div id="secProb"></div>
    <div id="secSmooth"></div>
    <div id="secPerp"></div>
    <div id="secMarkov"></div>
    <div id="secQuiz"></div>
  `;
  mountProb();
  mountSmooth();
  mountPerp();
  mountMarkov();
  mountQuiz();
}

document.addEventListener("DOMContentLoaded", initApp);

// ─── SECTION: SENTENCE PROBABILITY ─────────────────────────────────────────
function sampleRowHtml(samples, onclickFn, activeSentence) {
  const inCorpus = IN_CORPUS_SAMPLES.map(s => `<button class="sample-chip${s === activeSentence ? " active" : ""}" onclick="${onclickFn}('${s.replace(/'/g, "\\'")}')">${escapeHtml(s)}</button>`).join("");
  const oov = OOV_SAMPLES.map(s => `<button class="sample-chip oov${s === activeSentence ? " active" : ""}" onclick="${onclickFn}('${s.replace(/'/g, "\\'")}')" title="Not in the training text">${escapeHtml(s)} <span style="opacity:.55;">(new)</span></button>`).join("");
  return inCorpus + oov;
}

function mountProb() {
  const panel = document.getElementById("secProb");
  panel.innerHTML = `
    <div class="sv-eyebrow">The Chain Rule</div>
    <div class="sv-card">
      <div class="sv-card-title">How Do You Score A Whole Sentence?</div>
      <p class="sv-desc">A language model can rate how likely a full sentence is by multiplying together the probability of each word given the words before it.</p>
      <div class="sblock">
        <div class="sblock-lbl">The chain rule</div>
        <code class="sv-code" style="display:block;padding:8px;margin-top:6px;">P(w₁,w₂,...,wₙ) = P(w₁) · P(w₂|w₁) · P(w₃|w₁,w₂) · ... · P(wₙ|w₁...wₙ₋₁)</code>
      </div>
      <div class="sv-insight i-prob">💡 Computing the exact context for every word is expensive, so n-gram models approximate each term using just the previous N−1 words — the same idea from the N-grams page.</div>
    </div>

    <div class="sv-eyebrow">Try It</div>
    <div class="sv-card">
      <div class="sv-card-title">Sentence Probability Calculator</div>
      <p class="sv-desc">Pick a sentence (or type your own), choose N, and watch the probability get built word by word.</p>
      <div class="n-tabs" id="probNTabs"></div>
      <div class="sample-row" id="probSamples"></div>
      <input class="sv-input" id="probSentence" style="margin-bottom:14px;">
      <div id="probSteps"></div>
      <div class="prob-total-box" id="probTotal"></div>
      <div class="sv-insight" id="probInsight"></div>
    </div>
  `;
  document.getElementById("probNTabs").innerHTML = [1, 2, 3].map(n => {
    const sel = STATE.prob.n === n ? ` sel-${n}` : "";
    return `<button class="n-tab${sel}" onclick="setProbN(${n})">${N_META[n].label}</button>`;
  }).join("");
  document.getElementById("probSamples").innerHTML = sampleRowHtml(null, "setProbSentence", STATE.prob.sentence);
  const input = document.getElementById("probSentence");
  input.value = STATE.prob.sentence;
  input.addEventListener("input", e => { STATE.prob.sentence = e.target.value; renderProb(); });
  renderProb();
}

function setProbN(n) {
  STATE.prob.n = n;
  document.getElementById("probNTabs").innerHTML = [1, 2, 3].map(k => {
    const sel = STATE.prob.n === k ? ` sel-${k}` : "";
    return `<button class="n-tab${sel}" onclick="setProbN(${k})">${N_META[k].label}</button>`;
  }).join("");
  renderProb();
}
window.setProbN = setProbN;

function setProbSentence(s) {
  STATE.prob.sentence = s;
  document.getElementById("probSentence").value = s;
  document.getElementById("probSamples").innerHTML = sampleRowHtml(null, "setProbSentence", s);
  renderProb();
}
window.setProbSentence = setProbSentence;

function renderProb() {
  const n = STATE.prob.n;
  const tokens = tokenize(STATE.prob.sentence);
  const stepsEl = document.getElementById("probSteps");
  const totalEl = document.getElementById("probTotal");
  const insightEl = document.getElementById("probInsight");

  if (tokens.length < n) {
    stepsEl.innerHTML = "";
    totalEl.innerHTML = "";
    insightEl.className = "sv-insight i-warn";
    insightEl.textContent = `Type at least ${n} word${n > 1 ? "s" : ""} to compute a probability.`;
    return;
  }

  const { steps, total } = sentenceProbability(tokens, n, MODELS[n], 0, VOCAB_SIZE);

  stepsEl.innerHTML = steps.map(s => `
    <div class="prob-step">
      <div class="prob-step-word">${escapeHtml(s.word)}</div>
      <div class="prob-step-ctx">${n === 1 ? "no context (unigram)" : `after "${escapeHtml(s.context)}"`} — seen ${s.count}/${s.ctxTotal} times</div>
      <div class="prob-step-val" style="color:${s.p === 0 ? "var(--neg)" : "var(--prob)"};">${fmtProb(s.p)}</div>
    </div>`).join("");

  const chainStr = steps.map(s => `<span style="color:${s.p === 0 ? "var(--neg)" : "inherit"};">${fmtProb(s.p)}</span>`).join(" <span style='opacity:.5'>×</span> ");
  totalEl.innerHTML = `
    <div class="prob-total-lbl">How the total is calculated</div>
    <div style="font-family:var(--mono);font-size:12px;color:var(--color-text-secondary,#5a5950);margin-bottom:10px;word-break:break-word;line-height:1.8;">${chainStr}</div>
    <div class="prob-total-lbl">= P(sentence)</div>
    <div class="prob-total-val" style="color:${total === 0 ? "var(--neg)" : "var(--prob)"};">${total === 0 ? "0 (impossible!)" : total.toExponential(4)}</div>
  `;

  if (total === 0) {
    insightEl.className = "sv-insight i-warn";
    insightEl.innerHTML = "⚠️ At least one word pair in this sentence was never seen in training, so multiplying it in collapses the <em>whole sentence's</em> probability to zero — even though the rest of the sentence made sense. Head to <strong>Smoothing</strong> to see the fix.";
  } else {
    insightEl.className = "sv-insight i-prob";
    insightEl.innerHTML = "💡 Notice how quickly this number shrinks — even a short, perfectly normal sentence has a tiny probability, because it's the product of several probabilities each less than 1. That's why real systems work with log-probabilities instead.";
  }
}

// ─── SECTION: SMOOTHING ─────────────────────────────────────────────────
function mountSmooth() {
  const panel = document.getElementById("secSmooth");
  panel.innerHTML = `
    <div class="sv-eyebrow">Fixing The Zero Problem</div>
    <div class="sv-card">
      <div class="sv-card-title">Why Are We Smoothing At All?</div>
      <p class="sv-desc">You just saw it happen above: one word pair the model had never seen made the <em>entire sentence's</em> probability collapse to exactly zero — even though the rest of the sentence was perfectly ordinary English. That's clearly too harsh. An unseen pair should make a sentence <em>less likely</em>, not flatly impossible. Smoothing exists purely to fix that.</p>
      <div class="sv-insight i-warn">⚠️ Without smoothing: one unseen bigram = probability 0 for the whole sentence, no matter how good the rest of it is. That's the exact bug we're patching.</div>
    </div>

    <div class="sv-eyebrow">The Fix</div>
    <div class="sv-card">
      <div class="sv-card-title">Add-K (Laplace) Smoothing</div>
      <p class="sv-desc">The fix is simple: pretend every possible word has been seen a few extra times, even ones that never appeared. That way nothing ever hits exactly zero.</p>
      <div class="sblock">
        <div class="sblock-lbl">Smoothed probability</div>
        <code class="sv-code" style="display:block;padding:8px;margin-top:6px;">P(word|context) = ( count(context, word) + k ) / ( count(context) + k · V )</code>
        <div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);margin-top:6px;">V = vocabulary size (${VOCAB_SIZE} words here). k=1 is classic "Laplace" or "add-one" smoothing.</div>
      </div>
    </div>

    <div class="sv-eyebrow">Try It</div>
    <div class="sv-card">
      <div class="sv-card-title">Before &amp; After Smoothing</div>
      <p class="sv-desc">Try one of the "new" sentences below — words the corpus never saw together — and watch the fix happen live.</p>
      <div class="n-tabs" id="smoothNTabs"></div>
      <div class="sample-row" id="smoothSamples"></div>
      <input class="sv-input" id="smoothSentence" style="margin-bottom:14px;">
      <div class="sv-slider-row">
        <span style="font-size:12px;color:var(--color-text-secondary,#5a5950);min-width:70px;">k = </span>
        <input type="range" class="sv-slider" id="smoothK" min="0" max="2" step="0.1" value="1">
        <span class="sv-slider-val" id="smoothKVal">1.0</span>
      </div>
      <div id="smoothTable"></div>
      <div class="grid2" style="margin-top:12px;">
        <div class="prob-total-box">
          <div class="prob-total-lbl">Without smoothing</div>
          <div class="prob-total-val" id="smoothBefore"></div>
        </div>
        <div class="prob-total-box">
          <div class="prob-total-lbl">With smoothing</div>
          <div class="prob-total-val" id="smoothAfter"></div>
        </div>
      </div>
      <div class="sv-insight i-smooth" id="smoothInsight"></div>
    </div>
  `;
  document.getElementById("smoothNTabs").innerHTML = [2, 3].map(n => {
    const sel = STATE.smooth.n === n ? ` sel-${n}` : "";
    return `<button class="n-tab${sel}" onclick="setSmoothN(${n})">${N_META[n].label}</button>`;
  }).join("");
  document.getElementById("smoothSamples").innerHTML = sampleRowHtml(null, "setSmoothSentence", STATE.smooth.sentence);
  const input = document.getElementById("smoothSentence");
  input.value = STATE.smooth.sentence;
  input.addEventListener("input", e => { STATE.smooth.sentence = e.target.value; renderSmooth(); });
  const slider = document.getElementById("smoothK");
  slider.addEventListener("input", e => { STATE.smooth.k = parseFloat(e.target.value); document.getElementById("smoothKVal").textContent = STATE.smooth.k.toFixed(1); renderSmooth(); });
  renderSmooth();
}

function setSmoothN(n) {
  STATE.smooth.n = n;
  document.getElementById("smoothNTabs").innerHTML = [2, 3].map(k => {
    const sel = STATE.smooth.n === k ? ` sel-${k}` : "";
    return `<button class="n-tab${sel}" onclick="setSmoothN(${k})">${N_META[k].label}</button>`;
  }).join("");
  renderSmooth();
}
window.setSmoothN = setSmoothN;

function setSmoothSentence(s) {
  STATE.smooth.sentence = s;
  document.getElementById("smoothSentence").value = s;
  document.getElementById("smoothSamples").innerHTML = sampleRowHtml(null, "setSmoothSentence", s);
  renderSmooth();
}
window.setSmoothSentence = setSmoothSentence;

function renderSmooth() {
  const n = STATE.smooth.n;
  const k = STATE.smooth.k;
  const tokens = tokenize(STATE.smooth.sentence);
  const tableEl = document.getElementById("smoothTable");
  const beforeEl = document.getElementById("smoothBefore");
  const afterEl = document.getElementById("smoothAfter");
  const insightEl = document.getElementById("smoothInsight");

  if (tokens.length < n) {
    tableEl.innerHTML = "";
    beforeEl.textContent = "—";
    afterEl.textContent = "—";
    insightEl.className = "sv-insight i-warn";
    insightEl.textContent = `Type at least ${n} words.`;
    return;
  }

  const unsmoothed = sentenceProbability(tokens, n, MODELS[n], 0, VOCAB_SIZE);
  const smoothed = sentenceProbability(tokens, n, MODELS[n], k, VOCAB_SIZE);

  tableEl.innerHTML = `
    <div class="tbl-wrap"><div class="tbl-scroll">
      <table class="sv-table">
        <thead><tr><th>Word</th><th>Context</th><th>Raw count</th><th>Unsmoothed P</th><th>Smoothed P</th></tr></thead>
        <tbody>
          ${unsmoothed.steps.map((s, i) => {
            const fixed = s.p === 0 && smoothed.steps[i].p > 0;
            return `<tr class="${fixed ? "smooth-fixed" : ""}">
              <td class="sv-mono">${escapeHtml(s.word)}</td>
              <td style="font-size:11px;color:var(--color-text-tertiary,#8a8878);">${n === 1 ? "—" : escapeHtml(s.context)}</td>
              <td class="sv-mono">${s.count}</td>
              <td class="sv-mono" style="color:${s.p === 0 ? "var(--neg)" : "inherit"};">${fmtProb(s.p)}</td>
              <td class="sv-mono" style="color:var(--smooth);font-weight:700;">${fmtProb(smoothed.steps[i].p)}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div></div>
  `;

  beforeEl.textContent = unsmoothed.total === 0 ? "0 (impossible!)" : unsmoothed.total.toExponential(3);
  beforeEl.style.color = unsmoothed.total === 0 ? "var(--neg)" : "inherit";
  afterEl.textContent = smoothed.total.toExponential(3);
  afterEl.style.color = "var(--smooth)";

  const fixedCount = unsmoothed.steps.filter((s, i) => s.p === 0 && smoothed.steps[i].p > 0).length;
  insightEl.className = "sv-insight i-smooth";
  if (unsmoothed.total === 0 && smoothed.total > 0) {
    insightEl.innerHTML = `💡 Smoothing rescued ${fixedCount} word${fixedCount > 1 ? "s" : ""} that had zero probability — the sentence went from "impossible" to just "unlikely," which is a much more honest answer.`;
  } else {
    insightEl.innerHTML = `💡 This sentence didn't need rescuing — but notice smoothed probabilities are still a little different from the raw counts. That's the trade-off: smoothing takes a sliver of probability from things it <em>has</em> seen to cover things it hasn't.`;
  }
}

// ─── SECTION: PERPLEXITY ────────────────────────────────────────────────
function mountPerp() {
  const panel = document.getElementById("secPerp");
  panel.innerHTML = `
    <div class="sv-eyebrow">Scoring The Model</div>
    <div class="sv-card">
      <div class="sv-badge b-perp">Evaluation</div>
      <div class="sv-card-title">How Do You Know If A Language Model Is Good?</div>
      <p class="sv-desc">Perplexity turns a sentence's probability into a single score you can compare across models. Lower is better — it means the model was less "surprised" by the real text.</p>
      <div class="sblock">
        <div class="sblock-lbl">Perplexity</div>
        <code class="sv-code" style="display:block;padding:8px;margin-top:6px;">PP(W) = P(w₁,...,wₙ) ^ (−1/n)</code>
        <div style="font-size:11px;color:var(--color-text-tertiary,#8a8878);margin-top:6px;">Intuition: a perplexity of 20 means the model was, on average, as unsure as if it had to guess uniformly among 20 equally likely words at each step.</div>
      </div>
    </div>

    <div class="sv-eyebrow">Try It</div>
    <div class="sv-card">
      <div class="sv-card-title">Compare Unigram vs Bigram vs Trigram</div>
      <p class="sv-desc">Pick a sentence and see how confident each model order is about it (smoothing is on, k=1, so we always get a real number).</p>
      <div class="sample-row" id="perpSamples"></div>
      <input class="sv-input" id="perpSentence" style="margin-bottom:16px;">
      <div id="perpBars"></div>
      <div class="sv-insight i-perp" id="perpInsight"></div>
    </div>
  `;
  document.getElementById("perpSamples").innerHTML = sampleRowHtml(null, "setPerpSentence", STATE.perp.sentence);
  const input = document.getElementById("perpSentence");
  input.value = STATE.perp.sentence;
  input.addEventListener("input", e => { STATE.perp.sentence = e.target.value; renderPerp(); });
  renderPerp();
}

function setPerpSentence(s) {
  STATE.perp.sentence = s;
  document.getElementById("perpSentence").value = s;
  document.getElementById("perpSamples").innerHTML = sampleRowHtml(null, "setPerpSentence", s);
  renderPerp();
}
window.setPerpSentence = setPerpSentence;

function renderPerp() {
  const tokens = tokenize(STATE.perp.sentence);
  const barsEl = document.getElementById("perpBars");
  const insightEl = document.getElementById("perpInsight");

  if (tokens.length < 3) {
    barsEl.innerHTML = "";
    insightEl.className = "sv-insight i-warn";
    insightEl.textContent = "Type at least 3 words so all three model orders can be compared.";
    return;
  }

  const results = [1, 2, 3].map(n => {
    const { total } = sentenceProbability(tokens, n, MODELS[n], 1, VOCAB_SIZE);
    return { n, pp: perplexity(total, tokens.length) };
  });

  const maxPP = Math.max(...results.map(r => r.pp));
  const nColors = { 1: "var(--uni)", 2: "var(--bi)", 3: "var(--tri)" };

  barsEl.innerHTML = results.map(r => {
    const widthPct = Math.max(6, Math.round((r.pp / maxPP) * 100));
    return `
      <div class="perp-bar-row">
        <div class="perp-bar-label">${N_META[r.n].label}</div>
        <div class="perp-bar-track"><div class="perp-bar-fill" style="width:${widthPct}%;background:${nColors[r.n]};"><span>${r.pp.toFixed(1)}</span></div></div>
      </div>`;
  }).join("");

  const best = results.reduce((a, b) => (a.pp < b.pp ? a : b));
  const worst = results.reduce((a, b) => (a.pp > b.pp ? a : b));
  insightEl.innerHTML = best.n === worst.n
    ? `💡 All three orders scored about the same here.`
    : `💡 <strong>${N_META[best.n].label}</strong> got the lowest (best) perplexity at ${best.pp.toFixed(1)} — ${best.n > 1 ? "more context let it predict this sentence more confidently." : "in this case, ignoring context still worked out well, which won't always be true."} <strong>${N_META[worst.n].label}</strong> was the most "surprised," at ${worst.pp.toFixed(1)}.`;
}

// ─── SECTION: MARKOV MODELS ─────────────────────────────────────────────
const WEATHER_STATES = ["Sunny", "Cloudy", "Rainy"];
const WEATHER_ICON = { Sunny: "☀️", Cloudy: "☁️", Rainy: "🌧️" };
const WEATHER_TRANS = {
  Sunny:  { Sunny: 0.6, Cloudy: 0.3, Rainy: 0.1 },
  Cloudy: { Sunny: 0.3, Cloudy: 0.4, Rainy: 0.3 },
  Rainy:  { Sunny: 0.2, Cloudy: 0.3, Rainy: 0.5 },
};

function mountMarkov() {
  const panel = document.getElementById("secMarkov");
  panel.innerHTML = `
    <div class="sv-eyebrow">The Idea Behind It All</div>
    <div class="sv-card">
      <div class="sv-card-title">What's A Markov Model?</div>
      <p class="sv-desc">A Markov model describes a system that moves between a set of "states" over time. The key rule — the <strong>Markov property</strong> — is that the next state only depends on the <em>current</em> state, not on anything further back in history. The model has no memory beyond right now.</p>
      <div class="sblock">
        <div class="sblock-lbl">The Markov property</div>
        <code class="sv-code" style="display:block;padding:8px;margin-top:6px;">P(Xₜ₊₁ | Xₜ, Xₜ₋₁, ..., X₁) = P(Xₜ₊₁ | Xₜ)</code>
      </div>
      <div class="sv-insight i-markov">💡 This might sound familiar — a bigram language model <em>is</em> a Markov model where the "states" are words instead of weather. Predicting the next word from just the previous word is exactly this same assumption.</div>
    </div>

    <div class="sv-eyebrow">See It In Action</div>
    <div class="sv-card">
      <div class="sv-card-title">A Weather Forecast Machine</div>
      <p class="sv-desc">Three states — Sunny, Cloudy, Rainy — each with a probability of moving to any of the others tomorrow. Click a transition cell to inspect it, or generate a random forecast below.</p>
      <div class="mtx-wrap"><div class="mtx-grid" id="markovTransGrid" style="grid-template-columns:90px repeat(3, minmax(80px,1fr));"></div></div>
      <div class="sv-insight i-markov" id="markovTransInsight"></div>
    </div>

    <div class="sv-eyebrow">Try It</div>
    <div class="sv-card">
      <div class="sv-card-title">Generate A Random Forecast</div>
      <p class="sv-desc">Pick today's weather, then keep clicking "Next Day" — each new day is sampled using only the state you're currently in.</p>
      <div class="weather-pick-row" id="weatherPicks"></div>
      <div class="weather-row" id="weatherOutput"></div>
      <div style="display:flex;gap:8px;justify-content:flex-end;">
        <button class="sv-btn btn-sec" onclick="resetWeather()">↺ Reset</button>
        <button class="sv-btn" style="background:var(--markov);color:#fff;" id="weatherNextBtn" onclick="nextWeatherDay()" disabled>Next Day →</button>
      </div>
    </div>
  `;
  renderMarkovMatrix();
  renderWeatherPicks();
  renderWeatherOutput();
}

let markovSelectedTrans = null;

function selectMarkovTrans(from, to) {
  markovSelectedTrans = [from, to];
  renderMarkovMatrix();
}
window.selectMarkovTrans = selectMarkovTrans;

function renderMarkovMatrix() {
  const cells = [`<div class="mtx-cell mtx-head">today → tmrw</div>`];
  WEATHER_STATES.forEach(s => cells.push(`<div class="mtx-cell mtx-head">${WEATHER_ICON[s]} ${s}</div>`));
  WEATHER_STATES.forEach(from => {
    cells.push(`<div class="mtx-cell mtx-head">${WEATHER_ICON[from]} ${from}</div>`);
    WEATHER_STATES.forEach(to => {
      const v = WEATHER_TRANS[from][to];
      const sel = markovSelectedTrans && markovSelectedTrans[0] === from && markovSelectedTrans[1] === to;
      cells.push(`<button class="mtx-cell mtx-btn${sel ? " sel" : ""}" onclick="selectMarkovTrans('${from}','${to}')"><span style="color:${v >= 0.4 ? "var(--markov)" : "inherit"};font-weight:600;">${v.toFixed(1)}</span></button>`);
    });
  });
  document.getElementById("markovTransGrid").innerHTML = cells.join("");

  const insightEl = document.getElementById("markovTransInsight");
  if (!markovSelectedTrans) {
    insightEl.textContent = "💡 Click any cell to see that transition explained. Each row always adds up to 1.0 — from any state, something has to happen tomorrow.";
  } else {
    const [from, to] = markovSelectedTrans;
    const p = WEATHER_TRANS[from][to];
    insightEl.innerHTML = `💡 P(${to} | ${from}) = ${p.toFixed(1)} — if today is ${WEATHER_ICON[from]} ${from}, there's a ${Math.round(p * 100)}% chance tomorrow is ${WEATHER_ICON[to]} ${to}. Notice it doesn't matter what the weather was two days ago.`;
  }
}

function renderWeatherPicks() {
  document.getElementById("weatherPicks").innerHTML = WEATHER_STATES.map(s => `
    <button class="weather-pick${STATE.markov.sequence[0] === s ? " active" : ""}" onclick="startWeather('${s}')">
      ${WEATHER_ICON[s]}<div class="wp-lbl">${s}</div>
    </button>`).join("");
}

function startWeather(state) {
  STATE.markov.sequence = [state];
  STATE.markov.started = true;
  renderWeatherPicks();
  renderWeatherOutput();
  document.getElementById("weatherNextBtn").disabled = false;
}
window.startWeather = startWeather;

function nextWeatherDay() {
  const seq = STATE.markov.sequence;
  if (seq.length === 0) return;
  const current = seq[seq.length - 1];
  const probs = WEATHER_TRANS[current];
  const r = Math.random();
  let cumulative = 0;
  let next = WEATHER_STATES[WEATHER_STATES.length - 1];
  for (const s of WEATHER_STATES) {
    cumulative += probs[s];
    if (r <= cumulative) { next = s; break; }
  }
  seq.push(next);
  renderWeatherOutput();
}
window.nextWeatherDay = nextWeatherDay;

function resetWeather() {
  STATE.markov.sequence = [];
  STATE.markov.started = false;
  renderWeatherPicks();
  renderWeatherOutput();
  document.getElementById("weatherNextBtn").disabled = true;
}
window.resetWeather = resetWeather;

function renderWeatherOutput() {
  const seq = STATE.markov.sequence;
  const out = document.getElementById("weatherOutput");
  if (seq.length === 0) {
    out.innerHTML = `<span style="font-size:12px;color:#8a8878;">Pick today's weather above to begin.</span>`;
    return;
  }
  out.innerHTML = seq.map((s, i) => `<span class="weather-chip${i === 0 ? " start" : ""}" title="Day ${i + 1}: ${s}">${WEATHER_ICON[s]}</span>`).join("");
}

// ─── SECTION: QUIZ ──────────────────────────────────────────────────────
const QUIZ = [
  { id:1, tag:"Probability", q:"What rule lets us break P(sentence) down into smaller, per-word pieces?", opts:["The chain rule", "The Pythagorean theorem", "Bayes' theorem alone, with no other rule", "The law of large numbers"], ans:0, exp:"The chain rule of probability lets us write P(w₁,...,wₙ) as a product of conditional probabilities: P(w₁)·P(w₂|w₁)·P(w₃|w₁,w₂)... N-gram models then approximate each conditional using only the last N−1 words." },
  { id:2, tag:"Probability", q:"Why do real language models usually work with log-probabilities instead of raw probabilities?", opts:["Logs are required by law for statistics", "Multiplying many probabilities under 1 makes the number vanishingly small (underflow); logs turn multiplication into addition and stay numerically stable", "Log-probabilities are always positive numbers, which looks nicer", "It makes the sentence shorter to type"], ans:1, exp:"As you saw, even a 6-word sentence's probability can already be tiny (like 4×10⁻²). For long documents this underflows to 0.0 in floating point. Working in log-space avoids that, and turns the product into a sum: log P(a·b) = log P(a) + log P(b)." },
  { id:3, tag:"Probability", q:"A test sentence gets P(sentence) = 0 under a bigram model. What does that most likely mean?", opts:["The sentence is grammatically impossible in English", "At least one bigram in the sentence never occurred in the training data", "The model made a math error", "The sentence is too long to score"], ans:1, exp:"A probability of exactly zero almost always means sparsity — some specific word pair simply never showed up during training — not that the sentence itself is impossible or invalid. This is precisely the problem smoothing exists to fix." },
  { id:4, tag:"Smoothing", q:"What core problem does smoothing solve?", opts:["It solves zero probabilities for n-grams that were never seen in training", "It makes the model run faster", "It removes punctuation from sentences", "It increases the vocabulary size"], ans:0, exp:"Without smoothing, any unseen n-gram gets probability exactly zero, and one zero multiplies the entire sentence's probability down to zero too. Smoothing guarantees every possible word gets at least a tiny bit of probability." },
  { id:5, tag:"Smoothing", q:"In add-k (Laplace, when k=1) smoothing, what gets added to the denominator, and why?", opts:["Nothing — only the numerator changes", "k × V (V = vocabulary size), so the probabilities across the whole vocabulary still sum to 1", "The square root of the context count", "The total number of sentences in the corpus"], ans:1, exp:"Adding k to every possible word's count in a context means the context's total count grows by k·V (one k for each of the V vocabulary words). Adding that to the denominator keeps everything a valid probability distribution that still sums to 1." },
  { id:6, tag:"Smoothing", q:"What's a real downside of smoothing?", opts:["It has no downsides — it should always be used at maximum strength", "It takes some probability mass away from things the model actually observed, to cover things it hasn't — too much smoothing can flatten out real patterns", "It only works for unigram models", "It makes zero probabilities more common, not less"], ans:1, exp:"Smoothing is a trade-off: every bit of probability given to unseen events has to come from somewhere, and it comes from the events the model did observe. Too large a k over-flattens the distribution, making the model less confident even about things it has good evidence for." },
  { id:7, tag:"Perplexity", q:"Does a lower or higher perplexity indicate a better language model?", opts:["Lower is better", "Higher is better", "Perplexity doesn't relate to model quality", "It depends on the day of the week"], ans:0, exp:"Perplexity measures how \"surprised\" or confused a model is by the real text — lower means the model assigned the actual sequence higher probability, i.e. it predicted it more confidently and accurately." },
  { id:8, tag:"Perplexity", q:"How is perplexity mathematically related to a sentence's probability?", opts:["They are unrelated, independent numbers", "Perplexity = P(sentence)^(−1/N) — as probability goes up, perplexity goes down", "Perplexity is just probability multiplied by 100", "Perplexity is always exactly 1 divided by the vocabulary size"], ans:1, exp:"Perplexity is the inverse of the sentence's per-word probability, taken to the −1/N power (N = number of words). Higher P(sentence) always means lower perplexity, and vice versa — they move in opposite directions." },
  { id:9, tag:"Perplexity", q:"Why did the trigram model score lower (better) perplexity than the unigram model on the same sentence, as you saw in the comparison?", opts:["Trigram is always exactly 3 times better by definition", "More context (2 previous words instead of none) let it make sharper, more accurate predictions of what comes next", "The trigram model cheats by looking at future words", "It's random — there's no real reason"], ans:1, exp:"The unigram model ignores all context and just uses each word's overall frequency. The trigram model gets to condition on the previous two words, which — when there's enough training data — usually narrows down the likely next word far more precisely, lowering perplexity." },
  { id:10, tag:"Markov", q:"What is the \"Markov property\"?", opts:["The next state depends only on the current state, not on the full history before it", "Every state must eventually return to where it started", "All transition probabilities must be equal", "States can only move forward, never repeat"], ans:0, exp:"The Markov property (memorylessness) says P(next state | all previous states) = P(next state | current state only). Everything before the current state is irrelevant to what happens next." },
  { id:11, tag:"Markov", q:"Why are n-gram language models considered a type of Markov model?", opts:["They aren't — n-grams and Markov models are unrelated", "They assume the next word depends only on the previous N−1 words, not the entire sentence history — the same locality assumption as a Markov chain", "Because both were invented in the same year", "Because n-gram models use weather data"], ans:1, exp:"A bigram model literally is a first-order Markov chain where the \"states\" are words: P(next word | all previous words) is approximated as P(next word | just the last word). Higher-order n-grams are higher-order Markov chains." },
  { id:12, tag:"Markov", q:"In a Markov transition matrix, what must every row (all the outgoing probabilities from one state) add up to?", opts:["0", "1", "The number of states", "It varies row by row with no fixed rule"], ans:1, exp:"Each row lists the probabilities of moving from one specific state to every possible next state (including staying put). Since something must happen next, those probabilities have to sum to exactly 1 — you saw this pattern in the weather transition matrix." },
];

function quizTagInfo(tag) {
  const map = {
    Probability: { badge:"b-prob" },
    Smoothing:   { badge:"b-smooth" },
    Perplexity:  { badge:"b-perp" },
    Markov:      { badge:"b-markov" },
  };
  return map[tag] || { badge:"b-prob" };
}

function mountQuiz() {
  document.getElementById("secQuiz").innerHTML = `
    <div class="sv-eyebrow">Check Yourself</div>
    <div class="sv-card">
      <div class="sv-card-title">Concept Quiz</div>
      <p class="sv-desc">Two to three questions per topic. Pick an answer for instant feedback and a short explanation.</p>
      ${QUIZ.map((q, qi) => `
        ${qi > 0 ? '<hr class="qsep">' : ""}
        <div>
          <div style="display:flex;gap:8px;margin-bottom:10px;align-items:flex-start;">
            <span class="sv-badge ${quizTagInfo(q.tag).badge}" style="margin:2px 0 0;">${escapeHtml(q.tag)}</span>
            <div style="font-size:13.5px;font-weight:600;line-height:1.5;">${escapeHtml(q.q)}</div>
          </div>
          ${q.opts.map((opt, oi) => `<button class="qopt" id="lmqopt-${q.id}-${oi}" onclick="answerLmQuiz(${q.id},${oi},${q.ans})">${escapeHtml(opt)}</button>`).join("")}
          <div class="sv-insight" id="lmqfb-${q.id}" style="display:none;"></div>
        </div>`).join("")}
      <div style="text-align:center;margin-top:20px;">
        <button class="sv-btn btn-sec" onclick="resetLmQuiz()">↺ Reset quiz</button>
      </div>
    </div>
  `;
}

function answerLmQuiz(qid, oi, ansIdx) {
  const firstBtn = document.getElementById(`lmqopt-${qid}-0`);
  if (firstBtn.disabled) return;
  const q = QUIZ.find(x => x.id === qid);
  q.opts.forEach((opt, i) => {
    const btn = document.getElementById(`lmqopt-${qid}-${i}`);
    btn.disabled = true;
    if (i === ansIdx) btn.classList.add("correct");
    else if (i === oi) btn.classList.add("incorrect");
  });
  const fb = document.getElementById(`lmqfb-${qid}`);
  const correct = oi === ansIdx;
  fb.style.display = "block";
  fb.className = `sv-insight ${correct ? "i-pos" : "i-warn"}`;
  fb.innerHTML = `<strong>${correct ? "✓ Correct!" : "✗ Incorrect."}</strong> ${escapeHtml(q.exp)}`;
}
window.answerLmQuiz = answerLmQuiz;

function resetLmQuiz() { mountQuiz(); }
window.resetLmQuiz = resetLmQuiz;
