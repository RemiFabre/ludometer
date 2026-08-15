/* The settings panel: a gear, and what is behind it.
 *
 * Inline, never a pop-up — it opens by pushing the table down, the same way
 * everything else on this page reports itself. Animation speed is always here
 * (it exists because the alternative was letting the operating system decide —
 * see animate.js for that story); a page that shows score pop-ups asks for
 * their switch too with `{popups: true}`.
 *
 *   const settings = createSettings(host);                  // speed only
 *   const settings = createSettings(host, {popups: true});  // + score pop-ups
 *
 * Every control reads its stored value through its own module and writes it
 * back on every click, so a reload keeps whatever you chose.
 */

import { SPEEDS, initSpeed, prefersReducedMotion, setSpeed, speed } from "./animate.js";
import { initPopups, popupsOn, setPopups } from "./popups.js";
import { node } from "./dom.js";

const LABELS = { 0: "Off", 0.5: "0.5×", 1: "1×", 2: "2×" };
const TITLES = {
  0: "No tile animation: positions change at once",
  0.5: "Half speed — every tile easy to follow",
  1: "The default pace",
  2: "Twice as fast",
};

/* A gear, drawn rather than fetched — this page never asks the network for a
 * byte, not even an icon. Parsed as HTML so the SVG namespace comes for free
 * and this file need not name a URL of any kind. */
const GEAR_SVG =
  '<svg viewBox="0 0 24 24" class="gear-icon" aria-hidden="true"><path d="' +
  "M12 8.4a3.6 3.6 0 1 0 0 7.2 3.6 3.6 0 0 0 0-7.2zm9 4.9v-2.6l-2.4-.4a6.9 6.9 0 0 0-.8-1.9" +
  "l1.4-2-1.8-1.8-2 1.4a6.9 6.9 0 0 0-1.9-.8L13.1 3h-2.6l-.4 2.4a6.9 6.9 0 0 0-1.9.8l-2-1.4" +
  "-1.8 1.8 1.4 2a6.9 6.9 0 0 0-.8 1.9L3 10.7v2.6l2.4.4c.2.7.4 1.3.8 1.9l-1.4 2 1.8 1.8 2-1.4" +
  "c.6.4 1.2.6 1.9.8l.4 2.4h2.6l.4-2.4c.7-.2 1.3-.4 1.9-.8l2 1.4 1.8-1.8-1.4-2c.4-.6.6-1.2.8-1.9l2.4-.4z" +
  '"/></svg>';

function gearIcon() {
  const holder = document.createElement("span");
  holder.innerHTML = GEAR_SVG;
  return holder.firstChild;
}

export function createSettings(host, options = {}) {
  host.classList.add("settings");
  host.innerHTML = "";

  const head = node("div", "settings-head");
  const gear = node("button", "gear");
  gear.type = "button";
  gear.setAttribute("aria-expanded", "false");
  gear.appendChild(gearIcon());
  gear.appendChild(node("span", null, "Settings"));
  const summary = node("span", "settings-note", "");
  head.append(gear, summary);

  const panel = node("div", "settings-panel");
  panel.hidden = true;
  const label = node("span", "settings-label", "Tile animation");
  const speeds = node("div", "speeds");
  speeds.setAttribute("role", "group");
  speeds.setAttribute("aria-label", "Animation speed");
  const hint = node("p", "settings-hint", "");
  panel.append(label, speeds, hint);

  const buttons = SPEEDS.map((value) => {
    const b = node("button", "speed", LABELS[value]);
    b.type = "button";
    b.dataset.speed = String(value);
    b.title = TITLES[value];
    b.addEventListener("click", () => {
      setSpeed(value);
      sync();
      if (options.onChange) options.onChange(value);
    });
    speeds.appendChild(b);
    return b;
  });

  let popFlags = [];
  if (options.popups) {
    initPopups();
    panel.append(
      node("i", "settings-gap"),
      node("span", "settings-label", "Score pop-ups")
    );
    const flags = node("div", "speeds");
    flags.setAttribute("role", "group");
    flags.setAttribute("aria-label", "Score pop-ups");
    popFlags = [true, false].map((value) => {
      const b = node("button", "flag", value ? "On" : "Off");
      b.type = "button";
      b.dataset.pops = String(value);
      b.title = value
        ? "Every point floats off the square that earned it as the round is scored"
        : "Scoring stays in the panel below the boards";
      b.addEventListener("click", () => {
        setPopups(value);
        sync();
        if (options.onChange) options.onChange(speed());
      });
      flags.appendChild(b);
      return b;
    });
    panel.append(flags);
  }

  host.append(head, panel);

  function sync() {
    const now = speed();
    buttons.forEach((b) => {
      b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === now));
    });
    popFlags.forEach((b) => {
      b.setAttribute("aria-pressed", String((b.dataset.pops === "true") === popupsOn()));
    });
    const bits = [now ? "animation " + LABELS[now] : "animation off"];
    if (options.popups) bits.push(popupsOn() ? "pop-ups on" : "pop-ups off");
    summary.textContent = bits.join(" · ");
    hint.textContent = prefersReducedMotion()
      ? "Your system asks apps to reduce motion. The tiles still move here, because " +
        "this switch — not that one — decides. Pick Off if you would rather they did not."
      : "Tiles always travel to where they land. Slow it down to follow a long turn, " +
        "or turn it off for instant play.";
  }

  gear.addEventListener("click", () => {
    const opening = panel.hidden;
    panel.hidden = !opening;
    host.classList.toggle("open", opening);
    gear.setAttribute("aria-expanded", String(opening));
  });

  initSpeed();
  sync();

  return { el: host, sync, open: () => gear.click() };
}
