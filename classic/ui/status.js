/* The status band: one glance, whole game state.
 *
 * A single line of display type says what is happening right now ("Your turn —
 * pick a colour", "AI is thinking — 2.1s of 5s", "You won 74–68"), a second line
 * carries the detail, and the running score sits on the right. When the AI is
 * searching, the glaze bar under the text fills for its budget: the band is the
 * clock, so nothing has to cover the board to tell you to wait.
 *
 *   const status = createStatus(el);
 *   status.set({ headline: "Your turn", detail: "pick a colour", tone: "you" });
 *   status.startClock({ budget: 5, label: (s) => `AI is thinking — ${s}s` });
 */

import { node } from "./dom.js";

export function createStatus(host) {
  host.classList.add("status");
  host.setAttribute("role", "status");
  host.setAttribute("aria-live", "polite");

  const text = node("div", "status-text");
  const headline = node("p", "status-headline", "");
  const detail = node("p", "status-detail", "");
  text.append(headline, detail);

  const tally = node("div", "status-score");
  const you = scoreCell("You");
  const them = scoreCell("AI");
  tally.append(you.el, node("span", "status-dash", "–"), them.el);

  const bar = node("div", "status-bar");
  const fill = node("i", "status-fill");
  bar.appendChild(fill);

  host.append(text, tally, bar);

  let timer = null;

  function scoreCell(label) {
    const el = node("div", "status-side");
    const name = node("span", "status-side-name", label);
    const value = node("span", "status-side-score", "0");
    el.append(value, name);
    return { el, name, value };
  }

  function stopClock() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    host.classList.remove("ticking");
    fill.style.width = "0%";
  }

  return {
    el: host,
    /** Set the whole band. `tone` styles it: you / ai / scoring / end / idle. */
    set(next) {
      const info = next || {};
      headline.textContent = info.headline || "";
      detail.textContent = info.detail || "";
      host.dataset.tone = info.tone || "idle";
      if (!info.keepClock) stopClock();
    },
    /** Update the two-sided score readout. */
    setScore(left, right, leftLabel, rightLabel) {
      you.value.textContent = left;
      them.value.textContent = right;
      if (leftLabel) you.name.textContent = leftLabel;
      if (rightLabel) them.name.textContent = rightLabel;
      you.el.classList.toggle("leading", left > right);
      them.el.classList.toggle("leading", right > left);
    },
    /**
     * Run the clock in the band. `label(spent, budget)` writes the headline;
     * with a budget the glaze bar fills, without one it sweeps.
     */
    startClock(options) {
      const opts = options || {};
      const budget = Number(opts.budget) || 0;
      const started = Date.now();
      stopClock();
      host.classList.add("ticking");
      host.classList.toggle("indeterminate", !budget);
      const tick = () => {
        const spent = (Date.now() - started) / 1000;
        if (opts.label) headline.textContent = opts.label(spent, budget);
        if (budget) fill.style.width = Math.min(100, (spent / budget) * 100) + "%";
      };
      tick();
      timer = setInterval(tick, 100);
    },
    stopClock,
  };
}
