We have now followed one teacher decision all the way to a weight update. The final question is what those updates change when Kev plays a whole game. A lower loss tells us that it is fitting labels better. To see whether it can avoid ghosts and finish eating the pellets, we have to let its own decisions change the game.

Start with the loop across the top. We reset the native game to the same seed and starting level for every controller. Our reference suite has five seeds at each of levels one, two, three and five, giving twenty starts. We keep the game version, observation format, action rules and limits fixed. We also record the active checkpoint fingerprint, so a file called “latest” cannot quietly stand in for a different adapter.

In the first box, g sub t is the native game snapshot at decision t. The letter O is our observation function. It turns that snapshot into s sub t, the information Kev receives. This includes the full maze and all remaining pellets, power pellets, the player, the four ghosts, movement and native timers, along with the recent history from slide eight. The drawing is a diagram of the evaluation loop. It is not a cropped board fed to the model. The engine's hidden future randomness and the teacher's forecasts are not part of Kev's observation.

The middle box says a sub t equals pi sub phi of s sub t. Pi is simply a name for the learned policy: the rule that maps an observation to an action. Phi identifies the LoRA and pointer-head weights we just trained. A sub t is the direction it chooses. As we saw earlier, Kev scores the legal options and the controller executes the option with the largest probability. This compact notation leaves the fixed instruction and the legal option list implicit. Those inputs are still supplied in every real request.

During this run, the teacher does not rescue Kev. We can calculate what a teacher would choose or check the immediate risk of alternatives for diagnosis, but we do not replace the student's action. Otherwise we would be measuring a combined controller and could not tell whether Kev had learned the behavior.

The third box is the native game. Capital F names its transition function. Give it the current snapshot and chosen direction, and it produces g sub t plus one, the next snapshot. Here t counts decisions. One decision can advance several native frames before Pac-Man reaches the requested adjacent tile. The four ghosts keep their own controllers, the power timer keeps counting down, and collisions and pellet consumption follow the native mechanics. We observe the result and repeat the loop.

We stop at the first completed level or native game-over. Losing one life does not end the whole benchmark game if another life remains. A watchdog can also stop an unproductive run: too many decisions, too many simulation frames or five hundred and twelve decisions in a row without a pellet. That is an incomplete failure. It never becomes a win just because Pac-Man remained alive until a limit.

Now look at the two equations below. C is the clear rate. N is the number of games, twenty in this suite. The index j runs through those games. The expression one in square brackets is an indicator: it contributes one when game j genuinely clears the level, and zero otherwise. Add those indicators and divide by N. Watchdog exits stay in the denominator and contribute zero.

E measures food efficiency. P is the total number of pellets collected, including normal and power pellets. Lowercase n sub f is the total count of active native frames; the small f reminds us that these are frames. Divide P by that count, then multiply by a thousand to express the result as pellets per thousand frames. We aggregate the totals first. We do not average per-game ratios, and we do not put model or network waiting time into this denominator. If we want serving speed, we measure wall-clock latency separately.

The table is a real earlier result, from native-v3 and the frozen teacher on those same twenty starts. V3 clears zero games. The teacher clears all twenty. V3 has twenty-two immediate deaths where another legal move survives the next-tile transition; the teacher has none. That narrow check does not tell us whether a route was safe several decisions earlier. The teacher still loses one life in this reference suite before continuing to clear. Its earlier qualification suite used different seeds.

Food efficiency is about thirty-four point six for v3 and seventy-four point one for the teacher. V3 had improved some immediate retreats compared with v2, yet its food efficiency got worse and its longest dry spell reached two hundred and seventy-three decisions. That is why we inspect survival and food progress together. A player can avoid the next ghost while wandering away from the remaining pellets. A strict repeated-loop count can also miss a long unproductive route through many different tiles.

These numbers are historical v3 results, not results for the v4 checkpoint you just trained. Run the notebook's paired v3–v4 comparison to fill in that answer. It also probes the five hundred and twelve hard development requests and breaks them down by behavior. Those fixed-state diagnostics help explain the result. The full games remain the main evidence. Inspect power-expiry mistakes, recovery after respawn, dry intervals and the final few pellets in the saved trajectories.

There is one final limit to the claim. These twenty starts are excluded from training, but we have already studied their failures while designing v4. They are a reference benchmark, not an untouched blind test. For a stronger generalization claim, use new disjoint starts and report them separately. The same classic maze also cannot establish performance on unseen maze layouts.

Finish CP1 by keeping the completed adapter and pointer head, fingerprint, configuration, training curves, paired metrics and full traces together, with a verified Drive backup. Show one decision that improved or one failure that remains, and explain it from the actual state. The interactive game is still an ungraded way to explore the learned policy. Our conclusion should come from the reproducible game evidence.

Sources:
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/benchmark-spec.json
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/gameplay_benchmark.py
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/hard_case_evaluation.py
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/evaluation/v3-analysis-2026-10-07/README.md
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/evaluation/v3-analysis-2026-10-07/ghost-evasion-evidence.json
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/evaluation/v3-analysis-2026-10-07/pellet-efficiency-evidence.json
https://github.com/yxc20089/QPlusLearning/blob/b5aa381/labs/lab-01-kev-pacman/evaluation/v3-analysis-2026-10-07/teacher-reference.json
https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb
