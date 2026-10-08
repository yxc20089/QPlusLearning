We now have an action. Why can this output path be fast? The backbone still does the expensive reading. Kev changes what happens after that reading: the pointer head scores the supplied choices directly, so it can finish without writing an answer one token at a time.

Follow the top row. This drawing shows a new board state in our Qwen3.5 serving setup. The model first computes the state prefix, then continues with one question row containing the instruction and every option. Those are two stages of backbone computation. All K choices belong to that row; K is the number of options, not the number of separate backbone runs. Once the hidden vectors are available, the pointer head produces the scores, and the softmax and argmax we just saw choose a move.

Compare that with asking a conventional decoder to write an answer. It first reads the prompt. We call that the prefill pass. Prefill gives the distribution for the first answer token, y-one. Here y-m means answer token number m. M is the total number of tokens in the answer. Each later token depends on the tokens already generated, so the decoder has to continue in sequence. A four-token answer needs three extra decode passes after prefill. The decoder can reuse its cache during those passes, but it still has to compute the next token each time.

The equations below compare just the readouts, after we have the hidden vectors. R means an approximate count of multiplications in that readout. It is an operation count, rather than elapsed seconds.

For Kev, d is the hidden-vector width, and d-p is the smaller pointer width. They are two thousand five hundred and sixty and two hundred and fifty-six in our lab. We project one decide vector and K option vectors, giving the K-plus-one term. Each projection costs roughly d times d-p multiplications. Then we take K dot products, each with d-p entries. That gives the final K times d-p term.

For the conventional dense language-model head, V is the vocabulary size. Each answer token requires a projection from d hidden features to V vocabulary scores. Across M tokens, that is approximately M times d times V multiplications. The additional backbone decode passes are outside this equation; the lower row shows where they happen.

Try one token in the animation controls. There are now zero extra decode passes. A constrained single-token classifier narrows this comparison, and a custom classification head can also score choices directly. So the benefit we are explaining is specific: Kev provides a trained option-scoring head and a decision API, while this text-output route writes a multi-token answer. Actual response time still depends on the input length, hardware and serving setup.

LoRA changes the features the backbone learns. The pointer head and output format determine how we return the decision. Our API can serialize its result as JSON without asking Qwen to generate that JSON. We have now followed the complete inference path. Next, we will look at the training stages that created these weights.

Presenter reference: The pinned model loads a backbone without a vocabulary output head. Qwen3.5 hybrid serving on a cold, single-question request runs a state-prefix pass followed by one causal question row containing all the options. This differs from a literal single-call claim. A repeated identical state can reuse its prefix; successive Pac-Man board states usually differ, so cross-turn cache hits are not assumed. Long states, multiple questions or CUDA graph batching can split work further. The pointer multiplication sketch is (K+1)d d_p + Kd_p; biases, scaling and softmax are small omitted terms. The text sketch assumes M dense vocabulary projections, each dV, plus M−1 extra autoregressive backbone decode passes after prefill. Optimized subset logits or a custom choice head can change that baseline. The diagram is a dependency/count sketch, not a latency measurement. The illustrative y_1…y_4 boxes do not claim a particular tokenizer segmentation. API usage.output_tokens counts serialized response text, not generated tokens.

Source: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/model.py

API serialization: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/serve.py

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-speed.html
