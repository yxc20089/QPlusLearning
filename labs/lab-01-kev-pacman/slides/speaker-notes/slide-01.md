Picture Pac-Man at a junction. It can move Left, Up or Right. We give Kev the current board and those three choices, and ask it to pick the next move. Let's follow that decision across the diagram.

The board description includes the maze, pellets, ghosts, timers and recent movement. Together with our instruction and the legal moves, it becomes a sequence of input tokens. That's the x in the first formula.

Qwen processes that sequence using two kinds of weights. The teal block shows the pretrained weights, W-zero, which stay fixed. The orange block shows the changes learned through LoRA, delta W. Those changes sit inside selected layers of the backbone. We've drawn them as one pair of branches so we can see how they combine; the same idea applies at many projections across the model.

The formula underneath says that the backbone, f, turns the input x into hidden vectors, H. The small theta under f names its effective parameters: the fixed base weights with the LoRA updates applied. Each hidden vector is a set of numbers describing the input at a particular token position. On the next slide, we'll see which positions matter for this decision.

The pointer head reads the decision vector and a vector for each supplied move. It learns how to compare them and produces a score for each option. We call that collection of scores z.

Softmax turns those scores into the probabilities p on the right. These example values give Left the largest share, so the application executes Left. The values are illustrative; they describe preference among the supplied moves, not the chance of surviving the game.

After that move, the board changes and we repeat the process. Kev still runs the backbone, but it can return a choice directly from these scores. That avoids generating an answer one token at a time.

As we go into the lab, keep the colors in mind: the pretrained base stays fixed; training changes the LoRA updates and the pointer head.

Sources and presenter reference: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py ; https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md ; https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/cloud_runtime.py

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-architecture.html
