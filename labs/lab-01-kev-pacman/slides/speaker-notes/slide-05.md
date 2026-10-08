The pointer head has given us one score for each move. We now turn those scores into probabilities, and choose a move from them.

Look at the equation first. s is our board state. a-i means candidate action number i, such as Left, Up or Right. p of a-i given s is the probability Kev assigns to that action for this state, with our question and supplied options held fixed. The z-i in the numerator is the raw score for that action from the pointer head.

T is the temperature. In our freshly trained lab checkpoints, it is one, so dividing by T leaves each score unchanged. exp means the exponential function. It turns every score into a positive number. The denominator adds those positive numbers across all the options. The large sigma means “add them up”; j runs from one through K, where K is the number of options we supplied. Dividing by that shared total makes the probabilities add up to one. This whole calculation is called softmax.

Follow the example underneath. Left has a score of one, Up has zero, and Right has two. These are illustrative scores, rather than an actual game prediction. At temperature one, softmax gives approximately twenty-four point five percent to Left, nine percent to Up, and sixty-six point five percent to Right.

Now read the equation at the bottom. Argmax means “find the index with the largest value.” The winning index is i-star, and a-star is the action at that index. Here it is Right. Kev's Choice API uses the largest unrounded probability; it does not randomly sample a move from these percentages.

This probability tells us which supplied option the model prefers. It does not tell us that Pac-Man has a sixty-six percent chance of surviving. A dangerous move can still win if the model gives it the highest score. In this lab, the choices are the legal directions. Kev still has to choose one when they all look poor. That is why we evaluate complete games.

In the animation controls, try a smaller positive temperature. The probabilities become more concentrated. Try a larger one, and they become flatter. Right stays the winner because the score order has not changed. Temperature calibration can adjust those probabilities; it cannot repair a bad action ranking. Kev's published release fitted a temperature separately, but we do not inherit that fitted value when we train a new lab stage. We keep T at one here.

So Kev ends this decision with one probability per supplied move and one chosen action. Next, we can look at why producing this small output is faster than generating an answer token by token.

Presenter reference: Pinned PointerHead divides logits by temperature only in evaluation; training always sees T=1. Kev's inference paths apply softmax over the option dimension. The Choice API chooses the largest unrounded probability deterministically and rounds reported probabilities to four decimal places. An exact tie chooses the first supplied option. The separate API confidence field is a normalized concentration measure, not simply the winning probability; this slide shows option probabilities. Our Pac-Man request supplies all wall-legal directions, including dangerous directions, with no implicit wait, none, abstention or confidence fallback. Fresh checkpoint metadata defaults to temperature 1.0, and warm-starting a new stage does not inherit the previous fitted temperature. Loading a completed checkpoint for inference uses its saved temperature unless explicitly overridden. Positive scalar temperature preserves fixed-logit ordering. For numerical stability, implementations may subtract the maximum scaled score before exponentiating; this gives the same probabilities.

Source: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py

Choice selection: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/api.py

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-softmax.html
