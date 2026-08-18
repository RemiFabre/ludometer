---
title: "Faïence ingest"
emoji: "🧺"
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

The game collector behind [Faïence](https://huggingface.co/spaces/RemiFabre/faience),
a free browser implementation of the rules of Azul where you play against a
neural net that trained by self-play.

When a game ends on the playing page (and unless the player switched sharing
off in Settings), the page sends an anonymous record of it here: the moves,
the tiles that were dealt, which net played, and the final score. No name, no
account, no location. This process replays every submission in the real game
engine and keeps only records that reproduce their own recorded deals and
final score, then commits them in batches to the public dataset
[RemiFabre/faience-games](https://huggingface.co/datasets/RemiFabre/faience-games),
where they become training data.

No IP addresses or user agents are logged or stored, ever. `GET /stats` shows
the live counters. Source: `web/ingest/` in
[RemiFabre/ludometer](https://github.com/RemiFabre/ludometer).
