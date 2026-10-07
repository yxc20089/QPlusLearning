# Choosing a teacher for the native arcade game

The Berkeley projects are a useful algorithm vocabulary, but their simulator is
different from our pinned masonicGIT engine. We implement the ideas independently
and qualify them using complete native games. We do not import student solutions
or treat a published controller's results as results for this lab.

| Method | Useful role here | Limitation |
| --- | --- | --- |
| BFS / UCS / A* food search | Cached maze distances and routes toward remaining pellets, including tunnels. UCS can account for danger costs. | A shortest route can intersect a moving ghost. Searching all 244 food subsets is much larger than a local route problem. |
| Minimax with alpha-beta | Appropriate if ghosts can choose adversarial actions; pruning permits deeper search. | Our normal ghosts follow native targeting rules. An adversarial model can be unnecessarily conservative; pruning does not fix a weak evaluation function. |
| Expectimax | Model actual uncertain transitions, especially frightened-mode turns. | Berkeley's simplifying uniform distribution over ghost moves does not describe normal arcade chase/scatter. |
| Native rollout MPC | Enumerate legal first moves; simulate each under food-routing policies and independently sampled future randomness. Prefer survival, route progress and avoidance of repeated dry cycles. | A finite policy portfolio and horizon can miss a better escape. It requires full-game validation and is not an optimality proof. |
| Real-time MCTS / UCT | Allocate simulation effort to promising branches; a strong next candidate if the rollout portfolio fails. | Requires a native adapter, rollout/value design and its own qualification. We have not implemented or measured this candidate. |
| Symbolically guided MCTS | Add safety advice to search and rollouts. This motivates checking immediate native collisions separately from food utility. | The paper's temporal-logic/solver advice is stronger than our one-action check. We do not implement its proof machinery or inherit its guarantees. |
| Approximate Q-learning / learned value function | Learn longer-term danger and food-routing value from experience. | It first needs its own training and evaluation; it is not a ready-made expert for distillation. |

Berkeley explicitly describes short-horizon minimax thrashing beside an uneaten
dot. Its evaluation-function exercise addresses this behavior. Consequently,
changing the search name alone does not solve our recorded loops. The candidate
must make progress during actual play, including the last pellets.
See [CS188 Project 1](https://inst.eecs.berkeley.edu/~cs188/archive/fa24/projects/proj1/),
[Project 2](https://inst.eecs.berkeley.edu/~cs188/archive/fa24/projects/proj2/) and
[Project 3](https://inst.eecs.berkeley.edu/~cs188/archive/fa24/projects/proj3/).

Pepels, Winands and Lanctot published real-time MCTS for **Ms. Pac-Man**, and the
Maastricht group reports competition success. That supports investigating MCTS;
it does not establish performance in this different classic-Pac-Man engine.
Their reported enhancements include variable-depth search, informed player/ghost
rollouts, long-term objectives, and reuse of search trees with decayed statistics.
Those are useful design priorities for a native MCTS implementation, particularly
when replanning otherwise keeps promising food without reaching it.
See the [authors' project](https://project.dke.maastrichtuniversity.nl/games/games_pacman.htm)
and [paper record](https://cris.maastrichtuniversity.nl/en/publications/real-time-monte-carlo-tree-search-in-ms-pac-man/).

Busatto-Gaston, Chakraborty and Raskin study MCTS with symbolic advice in a
different Pac-Man environment. We borrow the separation of safety and reward as
a design idea, without claiming to reproduce their algorithm or reported scores.
See [Monte Carlo Tree Search guided by Symbolic Advice for MDPs](https://arxiv.org/abs/2006.04712).

Our first measured MPC development version cleared 20/20 boards with no deaths,
but produced many dry cycles. It was rejected. The qualification specification
is fixed before testing its separate reserved seeds. All boards must clear, no
avoidable immediate death or dry cycle is allowed, and a long pellet stall or
incorrect movement boundary fails qualification. Only a matching source/configuration
receipt with independently verified native replays can unlock new data creation.
Passing this finite suite is empirical evidence, never a universal safety claim.

The final development comparison uses the same 20 native initial boards (five
development seeds at levels 1, 2, 3, 5). Both reports include source hashes and
independently verified replay bundles. The MPC candidate is frozen before any
qualification game; development success alone does not unlock training.

| CPU controller | Maze clears | Lives lost | Avoidable immediate deaths | Dry-cycle decisions | Longest pellet stall |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Simple route heuristic](heuristic-development.json) | 16/20 | 32 | 18 | 77 | 117 decisions |
| [Native rollout MPC](teacher-development.json) | 20/20 | 0 | 0 | 0 | 92 decisions |

MPC collects all 4,880 pellets, including 80 power pellets, and eats 50 ghosts.
These are algorithm results, **not learned Kev results**. Four CPU workers took
about 13 minutes for the MPC development suite on the author's Mac; the heuristic
is much faster. This offline teacher cost is separate from LoRA training or
inference cost. The compact [MPC replays](teacher-development-replays.zip) and
[heuristic replays](heuristic-development-replays.zip) expose every native result.

For this lab, use BFS maze distances as the route primitive and the native engine
as the transition model. Normal targeting is predictable from the player and
ghost state; frightened turns require multiple independently sampled futures.
Rank immediate survival and dry-cycle avoidance separately from food progress.
Predict time to the next actual pellet consumption rather than only distance
to a pellet: pixel offsets, eating pauses, moving ghosts and replanning can make
those disagree. Full-maze completion tests this potential beyond a few good
opening moves. If the fixed rollout portfolio fails, add native MCTS and measure
it under the same reserved suite before distillation; do not substitute a
competition score or a short successful clip for qualification.
