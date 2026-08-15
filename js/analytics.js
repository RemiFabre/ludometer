/* An honest, cookie-free tally of who plays — OFF until an endpoint is named.
 *
 * The page promises "nothing is sent anywhere", and while COUNT_URL below is
 * empty that stays literally true: this module produces zero network traffic
 * and the rest of the page never has to think about it.
 *
 * To switch it on, create a (free) GoatCounter account and put its count URL
 * here, e.g. "https://ludometer.goatcounter.com/count". GoatCounter is built
 * for exactly this arrangement: no cookies, no fingerprinting, no personal
 * data, nothing stored in the visitor's browser — so no consent banner is
 * needed — and the dashboard shows visits plus each named event below. When it
 * is on, app.js appends one sentence to the "How this works" panel saying so;
 * the page never claims silence it does not keep.
 *
 * What gets counted (each a single GET of a 1×1 gif, fired and forgotten):
 *   pageview      the page was opened
 *   game-start    tiles were dealt
 *   game-win      the human beat the net
 *   game-loss     the net won
 *   game-draw     a draw
 */

export const COUNT_URL = ""; // e.g. "https://ludometer.goatcounter.com/count"

/** Whether the tally is on at all — the About panel reads this. */
export const analyticsOn = () => !!COUNT_URL;

/**
 * Count one event. Never throws, never waits, does nothing while COUNT_URL is
 * empty. `pageview` is special-cased to count as a visit rather than an event.
 */
export function track(name) {
  if (!COUNT_URL) return;
  try {
    const page = name === "pageview";
    const path = page ? location.pathname : name;
    const url =
      COUNT_URL +
      "?p=" + encodeURIComponent(path) +
      (page ? "" : "&e=true") +
      "&rnd=" + Date.now().toString(36);
    new Image().src = url; // a GET the browser fires and forgets; no reply is read
  } catch (err) {
    /* an ad blocker or a locked-down browser; the game plays on */
  }
}
