---
title: "Faïence"
emoji: "🀄"
colorFrom: blue
colorTo: yellow
sdk: static
pinned: true
thumbnail: https://remifabre-faience.static.hf.space/social.png
short_description: "Azul rules vs a self-play neural net, all in your browser"
---

<!-- The card for the Faïence Space. This file lives in the ludometer repo at
     web/player/space/README.md; scripts/deploy_player.sh copies it to the
     Space root at deploy time (the space/ directory itself is never shipped
     to visitors). -->

Faïence is a free, open-source implementation of the rules of *Azul*, the
tile-laying game by Michael Kiesling: a fan project for research, with its own
code and artwork, not affiliated with or endorsed by the game's publishers.

Your opponent is a neural network that learned the game from scratch by
self-play. The net and its tree search run entirely in your browser: your
browser downloads the model once and everything after that is local. No
server plays for it, no account, no ads, nothing to buy.

It began as a machine-learning experiment: Ludometer, a research framework
that measures how good a board game is from the shape of an AI's learning
curve. The games people play here are the research material: when a game
ends, the page sends an anonymous record (moves, deals, net, score; sharing
can be switched off in Settings) to the public dataset
[RemiFabre/faience-games](https://huggingface.co/datasets/RemiFabre/faience-games),
where it becomes training data.

Code, training logs and methodology:
[RemiFabre/ludometer](https://github.com/RemiFabre/ludometer) (MIT).
