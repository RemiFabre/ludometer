/* In the repo this is a shim so the ingest can be developed and tested against
 * the one true engine. At deploy time scripts/deploy_player.sh REPLACES this
 * file with a copy of web/player/js/engine.js, so the Space is self-contained
 * and validates games with byte-for-byte the same rules the page plays by. */
export * from "../player/js/engine.js";
