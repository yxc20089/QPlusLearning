Imagine Pac-Man has reached a junction. Left, Up and Right are legal moves. We want Kev to choose one of those moves from the board we have right now. Follow the diagram from left to right, or open the animation controls and step through it with me.

On the left, x is our input token IDs. The input contains the maze, pellets and power pellets, ghost information, relevant timers and movement history, the decision instruction, and the moves we allow. The three moves here are an illustration, not a restriction to three actions throughout the game.

Inside the backbone, teal means fixed and orange means trainable. W-zero is the collection of pretrained base weights. Delta W means the change learned through LoRA. The original base weights stay fixed, while LoRA adds learned updates to selected projections inside many layers. We have drawn one pair of branches to make that relationship visible; we are not adding a single adapter after the whole backbone. Our lab uses rank 16. We will unpack what that rank means shortly.

Read the first formula as: H equals the backbone applied to x, using the base weights together with the LoRA updates. The letter f stands for the backbone network. The semicolon separates the input from its weights. Capital H collects the contextual hidden vectors produced by the network. W-zero plus delta W is shorthand for the adapted projections, not an update to every weight in Qwen.

Next is the pointer head, which also learns. It reads the hidden vector at the final decision marker and the vectors at the closing markers of the supplied options. Two learned projections turn those vectors into one score for each move. Lowercase z is the vector of these scores. Our lab's pointer projections have dimension 256. We will draw the exact dot-product calculation on its own slide.

The second formula says that softmax converts z into p. Lowercase p is a probability distribution over our supplied moves. For this illustration only, scores of two, one and zero give about 0.67, 0.24 and 0.09. These are not measurements from our trained model, and 0.67 does not mean a 67 percent chance of surviving the game.

Left has the highest probability here, so the application selects that label and executes the move. The next board gives us the next decision input. Kev still computes the backbone representations; its speed advantage is that it scores supplied options directly, without generating an answer or JSON response one token at a time.

Before moving on, point to the fixed base weights, then to the two things training changes: the LoRA updates and the pointer head.

Implementation sources: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py ; https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md ; https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/cloud_runtime.py
Animation controls: https://yxc20089.github.io/QPlusLearning/animations/kev-architecture.html
