Let's zoom into the input we just gave Kev. There are three ingredients: the state, the instruction, and the moves it is allowed to choose.

The state, s, describes the board now. In our lab, that includes the remaining pellets and power pellets, Pac-Man's position and heading, each ghost's state, timers and recent movement. The instruction, u, tells Kev what decision to make. A is the ordered list of legal moves. For this junction, we're using Left, Up and Right.

The first formula says that the encoder combines those ingredients into x, our input token sequence. The boxes here group text spans; a phrase such as Left may take one or more tokens. Kev inserts special boundary tokens around those spans. We show them with readable names like state, question and option.

Look at the closing marker after each move. That's the position where Kev will read that option's hidden vector. Because the backbone processes the sequence causally, the closing marker can use the option text that came before it. With this lab's configuration, later options can also see earlier options.

Now look at the final decide marker. It comes after all three choices, so its hidden vector can use the board, the instruction and every supplied option. That gives the pointer head a place to read the whole decision context.

Qwen produces a hidden vector at every token position. All those vectors together form H. The notation at the bottom says H has L rows and d columns: L is the number of input tokens, and d is the width of each hidden vector. Little h means one vector from that collection. The subscripts identify the closing marker for option one, two or three, or the final decide marker.

The highlighted four vectors are the ones we pass to the pointer head for this three-option example. We'll work through that comparison shortly. First, let's see how LoRA changes the representations those vectors contain.

Presenter reference: readable markers map to existing Qwen tokens: <state> → <|fim_prefix|>, <q> → <|fim_middle|>, <opt> → <|box_start|>, </opt> → <|box_end|>, <decide> → <|fim_suffix|>. No new embedding rows are required. The lab uses Qwen3.5's causal row form with option_isolation=false. The slide's state, instruction and option text is shortened for illustration; it is not an exact request dump or a literal token count.

Source: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py
Lab state: https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/games/arcade-engine.js
Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-input.html
