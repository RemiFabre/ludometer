# The ideas that moved this project

*A short ledger of what actually made a difference, in the order it
happened, for a future write-up. Public-facing: it says what was done, not
where every byte came from.*

1. **Learning curves as a measure of a game.** The project's thesis: train
   the same self-play learner on several games and compare how Elo grows
   with practice; a game with a long, steady curve has depth. Azul was the
   calibration case, Uno the contrast, Connect Four and Lost Cities the
   controls (`docs/METHODOLOGY.md`, `docs/NEXT_GAMES.md`).
2. **A fixed anchor ladder.** Every checkpoint is rated against the same
   frozen opponents (random = 0), so curves are comparable across runs and
   months.
3. **AlphaZero-style self-play with a structured net.** Entity attention
   over factories, centre, lines and walls instead of a flat MLP; a margin
   head so the net plays to win by a lot once winning is certain; chance
   nodes by re-sampled determinization for the refill.
4. **Batched self-play.** Many games per process, one forward pass for all
   their leaves: the net stops being the bottleneck.
5. **Wall-clock honesty.** The only Elo that counts for the browser is
   measured at matched think time on a laptop CPU. It killed two "stronger"
   nets that were only stronger at equal search, and it set the bar for
   every release since (+150 to take the top slot).
6. **Expert games.** A few thousand games of top-level human play were
   collected and replayed through the engine (every game validated by
   reaching the reported final score). Fine-tuning the strongest net on the
   experts' moves, with its own games as rehearsal, made it measurably
   stronger; the effect concentrated in the early, strategic part of the
   game.
7. **Teacher and student.** The strongest net was too slow to ship, so it
   became a teacher: it played itself with long searches on a fleet of
   cheap cloud machines, and a smaller, faster student learned its
   *searched* opinions (visit distributions and root values), plus every
   position of the expert games searched the same way. Copying searched
   opinions gave +200 Elo where copying raw outputs had given +20.
8. **The search's value as a target.** Half game outcome, half the root
   value the search itself computed: a less noisy signal.
9. **A fleet, not a supercomputer.** Dozens of small cloud jobs, weights
   and games moving through a hub; the laptop trains, the fleet plays.
   Later, the engine was ported to Rust (17× per GPU job), which turned the
   cost of a corpus from tens of dollars into a few.
10. **A net cannot be its own stronger teacher.** Distilling a net's own
    longer search gives parity; self-play polish gives ~+100 and stops. The
    next level always needs a teacher that is genuinely stronger at equal
    search.
11. **Experimental opponents.** A net that beats the champion but not by
    the bar ships behind a switch, never as the default.
12. *(open)* **Strategy from the experts.** The standing hypothesis: humans
    still out-plan the net in the first rounds and lose to its calculation
    later. `docs/STRATEGY_TASKFORCE.md`.
