/* An honest, cookie-free tally of who plays.
 *
 * GoatCounter is built for exactly this arrangement: no cookies, no
 * fingerprinting, no personal data, nothing stored in the visitor's browser —
 * so no consent banner is needed — and the dashboard is public, so a player
 * can read everything the page records; the About panel links straight to it.
 * The pings go to GoatCounter's /count endpoint directly (their count.js is
 * never loaded: this page ships every byte it runs). Empty COUNT_URL and the
 * module produces zero network traffic; app.js then also removes the About
 * panel's tally note and the stats button, so the page never points at a
 * tally it does not keep.
 *
 * What gets counted (each a single GET of a 1×1 gif, fired and forgotten):
 *   pageview                              the page was opened
 *   game-start                            tiles were dealt
 *   game-end/<model>/<result>             how a finished game went, per net —
 *                                         result is human-wins | net-wins | draw,
 *                                         with the score line as the hit's title
 *
 * The dashboard should be made PUBLIC in GoatCounter's settings: the page links
 * to it, so every player can see exactly what is recorded. That link appears
 * automatically once COUNT_URL is set.
 */

export const COUNT_URL = "https://faience.goatcounter.com/count";

/** Whether the tally is on at all — the About panel reads this. */
export const analyticsOn = () => !!COUNT_URL;

/** The public dashboard the About panel links to — the /count host itself. */
export function statsUrl() {
  if (!COUNT_URL) return "";
  try {
    return new URL(COUNT_URL).origin;
  } catch (err) {
    return "";
  }
}

/**
 * Count one event. Never throws, never waits, does nothing while COUNT_URL is
 * empty. `pageview` is special-cased to count as a visit rather than an event;
 * `extra.title` becomes the hit's title (detail on the dashboard, e.g. a score
 * line), while the path stays coarse so the counts aggregate.
 */
export function track(name, extra) {
  if (!COUNT_URL) return;
  try {
    // development, the headless test suite and any staging copy of the site
    // play thousands of games; none of them are players, and none of them may
    // touch the public tally
    const host = location.hostname;
    if (host === "localhost" || host === "127.0.0.1" || host === "" || host.includes("staging")) return;
    const page = name === "pageview";
    const path = page ? location.pathname : name;
    const url =
      COUNT_URL +
      "?p=" + encodeURIComponent(path) +
      (page ? "" : "&e=true") +
      (extra && extra.title ? "&t=" + encodeURIComponent(extra.title) : "") +
      "&rnd=" + Date.now().toString(36);
    new Image().src = url; // a GET the browser fires and forgets; no reply is read
  } catch (err) {
    /* an ad blocker or a locked-down browser; the game plays on */
  }
}
