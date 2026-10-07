On the last slide, we saw Qwen turn our board and choices into hidden vectors. LoRA lets training change those representations while keeping the pretrained base weights fixed. Let's look inside one of the projections where that happens.

The vector coming in on the left is v. It's an activation inside a layer, rather than the input token sequence x we used earlier. The output of this projection is y.

Follow the teal path first. The base matrix, W-zero, multiplies v to produce the original projection. That matrix has d-out rows and d-in columns. Those names simply mean the output width and the input width. Training keeps every value in this base matrix fixed.

Now follow the orange path. A maps the same input into a much smaller space with r features. B maps those r features back to the output width. Together, they produce B times A times v. We multiply that contribution by alpha divided by r, then add it to the base output. That's the first equation at the bottom.

The second equation names the resulting weight update, delta W. It is alpha over r times B times A. The order matters: A acts first, and B acts second. Their product can have rank at most r. That's where the name low-rank adaptation comes from.

In our Kev recipe, r is sixteen and alpha is thirty-two, so the multiplier is two. Sixteen is the width of this small intermediate space. It is not the number of layers we train, or the number of moves Kev can choose.

For one projection, A and B contain r times the sum of d-in and d-out parameters. A full update would need d-in times d-out. That makes the learned update much smaller when r is small relative to those widths.

Kev applies this idea across the attention projections, the DeltaNet projections in Qwen3.5's linear-attention layers, and the feed-forward projections. The all-targets setting includes those groups. Across our four-billion-parameter backbone, the LoRA matrices and pointer head together have about thirty-three point eight million trainable parameters. The base weights stay fixed, but we still run the backbone and backpropagate through it to learn the adapters.

The recipe also uses five percent dropout on the LoRA input during training. Evaluation turns that dropout off. With the usual initialization, A starts random and B starts at zero, so a new adapter begins with zero delta W. When we continue from v2 to v3, we load the learned adapter and head instead of resetting them.

So the hidden vectors can change as the adapter learns. On the next slide, we'll use those vectors to work through the pointer head's comparison of our supplied moves.

Presenter reference: A and B here are LoRA matrices, distinct from the ordered action list A on slide 2. The drawing shows one adapted linear projection. The scalar alpha/r applies to the LoRA contribution. Kev's pinned configuration uses rank 16, lora_alpha=32, lora_dropout=0.05 and lora_targets=all. Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b and out_proj. Special embedding training is disabled in this lab. The 33.8M count includes the pointer head and is not the count for one projection.

Sources: https://arxiv.org/abs/2106.09685 ; https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py ; https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/cloud_runtime.py

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-lora.html
