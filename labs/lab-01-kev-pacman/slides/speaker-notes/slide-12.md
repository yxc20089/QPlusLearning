# Slide 12 — How gradients update Kev

On the last slide, the teacher gave us Left and Kev assigned it a probability of 0.20. The loss told us that this was a poor match. Now we need to turn that mismatch into a learning signal.

Start with the equation at the top. Script L is our cross-entropy loss. z sub i is the raw score for action i, before softmax. The two partial-derivative symbols ask how the loss would change if we nudged that score while holding the other scores fixed. The answer is p sub i minus t sub i. p is Kev’s probability for that action. t is the target indicator: it is one for the teacher’s chosen action and zero for every other supplied action.

Using our illustrative probabilities, Up gives 0.35 minus zero, so its score gradient is positive 0.35. Left gives 0.20 minus one, so its gradient is negative 0.80. Right gives positive 0.45. A downhill direction in score space would raise Left’s score and lower the others. These numbers explain the derivative. They are not measured predictions from a trained checkpoint, and we are not directly editing scores during training.

Now follow the drawing from left to right once. x is the encoded state, question and action options. The backbone produces H, the collection of hidden vectors. The pointer head reads the decide marker and each option’s closing marker, then produces z, the list of action scores. Softmax and the teacher label give us the loss on the right. The backbone box is a schematic of many layers.

The orange arrows run in the reverse direction. First the loss reaches the pointer head. W sub q and b sub q are its query projection matrix and bias. W sub k and b sub k are its key projection matrix and bias. Autograd uses the chain rule through the dot products and both projections. It computes gradients for those head weights and for the hidden vectors that supplied their inputs. These head projections are separate from the attention projections inside Qwen.

Those hidden-vector gradients continue through the backbone. W-zero stays fixed, but the activations around it still depend on earlier layers. Freezing the matrix means we do not update its entries. It does not remove that operation from the path needed to learn an earlier adapter. This is why LoRA training still needs a backward pass through a large model. Our gradient checkpointing saves activation memory by recomputing selected forward operations during backward.

Inside each selected projection, A sub ell and B sub ell are the two LoRA matrices. Ell names the adapted projection. We have many such pairs across Qwen’s attention, DeltaNet and feed-forward projections. Each pair learns its contribution to the representation. Rank is sixteen, and alpha over rank is thirty-two over sixteen, which gives a multiplier of two. The original base matrices and token embeddings remain fixed in this lab.

Look at phi in the lower equation. The braces collect the parameters that training can update: every adapted layer’s A and B, together with the pointer head’s two matrices and two biases. That collection is what we optimize. The teacher’s search algorithm and the game simulator are outside this gradient path. Their role was to produce and check our fixed labels.

Finally, connect this picture to the notebook. We use microbatches of four requests and accumulate across two of them, so a full optimizer update combines eight source requests. Kev weights the losses by source records. It clips the global trainable gradient norm to one, then fused AdamW updates phi. The learning-rate schedule advances after that update, with a configured maximum of two times ten to the minus five. AdamW keeps running estimates of gradients and squared gradients, and applies weight decay. It is more than simply subtracting the three score gradients shown at the top.

The new v4 stage copies the completed v3 adapter and head, then starts a fresh optimizer. Recovery within v4 restores that stage’s optimizer and progress. Because each update combines examples and changes shared weights, one particular action probability need not improve at every step. On the next slide, we will test the resulting policy in complete games, where ghost avoidance and pellet efficiency matter.

## Sources and scope

- [Pinned Kev model](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py): affine pointer projections, readouts, all-targets rank-16 LoRA and trainable-parameter filtering.
- [Pinned Kev training loop](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/train.py): ordinary choice cross-entropy, record-weighted accumulation, global gradient clipping at 1.0, AdamW and OneCycleLR.
- [Lab execution overlay](https://github.com/yxc20089/QPlusLearning/blob/5b2db99/labs/lab-01-kev-pacman/training_monitor.py): CUDA fused AdamW with the upstream objective retained.
- [V4 notebook recipe](https://github.com/yxc20089/QPlusLearning/blob/5efd612/labs/lab-01-kev-pacman/notebooks/pacman_kev_v4.ipynb): completed-v3 warm start, fresh optimizer, batch 4 × accumulation 2, maximum learning rate 2e-5 and checkpointing.
- [LoRA paper](https://arxiv.org/abs/2106.09685): fixed pretrained matrices and learned low-rank updates.

The score derivative is an analytic consequence of single-label softmax cross-entropy at training temperature 1. It is displayed for one illustrative Pac-Man choice. The actual optimizer combines request losses and replay; this animation does not run GPU training, differentiate the teacher, reproduce a checkpoint’s predictions or establish better gameplay. A and B here are LoRA matrices, distinct from the action list A used earlier. The backbone drawing does not assert that there is only one layer or that every original parameter trains.
