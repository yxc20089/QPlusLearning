LoRA has adapted the hidden vectors. Now we need to compare the moves we supplied. The pointer head takes the readouts we picked earlier and turns them into one score for each move.

Start at the top left. h-decide is the hidden vector at the final decide marker. At that point, Qwen has read the board, our instruction and every candidate. We pass this vector through a learned linear layer to get q, the query vector. W-q is that layer's weight matrix, and b-q is its bias. We reuse this same query when we compare all the options for this question.

Now follow the lower path. h-end-i is the hidden vector at the closing marker for option i. The i simply identifies which option we're considering: Left, Up or Right in our junction example. This is the contextual vector at the marker, rather than an average of the option's token vectors.

A second learned linear layer maps that readout to k-i, the option's key vector. Its matrix is W-k and its bias is b-k. We apply this same layer to every option. Each option gets its own key because it has a different hidden vector, while the layer's weights are shared.

The d at the bottom names the width of the original hidden vectors. For our Qwen3.5 four-billion-parameter model, that width is two thousand five hundred and sixty. The pointer width, d-p, is two hundred and fifty-six. So each projection turns two thousand five hundred and sixty features into two hundred and fifty-six.

At the dot in the drawing, we compare q with k-i. Multiply their first entries, multiply their second entries, and keep going through all two hundred and fifty-six entries. Add those products. That is the dot product, written q-transpose times k-i in the equation. The transpose symbol tells us to treat q as a row when we multiply the two vectors.

Kev then divides the result by the square root of d-p. The square root of two hundred and fifty-six is sixteen, so that's the fixed divisor in our lab. The result is z-i: one raw score for option i. Kev uses an ordinary dot product here, without normalizing the vectors to unit length for a cosine similarity.

Repeat that comparison for each of the K legal options. K is just the number of choices we supplied. Three moves give us three scores. Both of these pointer projection layers train alongside the LoRA adapters, so the model can learn which decision features should match which option features. The head is scoring the moves; the teacher's future-game simulations happen when we prepare the training labels.

We now have a score for each move. On the next slide, we'll turn those scores into probabilities and see how Kev selects an action.

Presenter reference: Pinned PointerHead uses two ordinary nn.Linear(d, dp) layers with biases and scale 1/sqrt(dp). W_q and W_k each have shape 256 × 2560; each bias has 256 entries, giving 1,311,232 head parameters. These are pointer-head parameters; Qwen's attention q_proj/k_proj are separate backbone projections. The head readouts are h[decide_idx] and h[opt_idx], where option indices point to closing markers. There is no value projection, attention-weighted value sum, MLP or cosine normalization in the pointer head. In the lab temperature is 1.0. The diagram shows one option i; the same query and key layer are used for the other options. No independence or option-order invariance is implied: the Qwen3.5 row is causal and option isolation is disabled.

Source: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-pointer.html
