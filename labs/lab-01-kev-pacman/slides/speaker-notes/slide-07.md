We have followed a decision all the way from input to action. Now let’s see where those learned weights came from. The arrows across the top show the general curriculum. Each stage starts with the LoRA and pointer head saved by the previous stage, then adds another set of decision examples.

We begin with Qwen3.5-4B-Base, whose language weights have already been pretrained. Our initial stage trains a fresh LoRA adapter and pointer head on twelve thousand five hundred and seventy-six generic requests, for two epochs. So when we call this initial decision training, we mean learning the decision adapter and head. We are not pretraining Qwen’s language weights again.

Next come dates and missing-evidence examples. These teach the model to use supplied day counts and recognize when the evidence needed for a decision is absent. Then come real consumer-finance documents, followed by harder skill selection and developer-tool decisions. The numbers below the boxes show the new requests plus generic replay. Replay means we mix in earlier decision examples while learning the new task. That gives the model practice on both, although it cannot guarantee that every earlier ability is retained.

Skills is the end of this general curriculum and the baseline for our game. The Skills box appears again in the lower row to show that handoff. It has learned how to score supplied options, but it has not yet received Pac-Man training labels. That is why seeing it play a game is useful: we can watch what general decision training transfers, and what it does not.

Native Pac-Man v2 starts from Skills. Its task labels come from the qualified teacher, whose game simulations we will examine shortly. Native v3 starts from v2 instead. It adds verified corrections at states reached by the v2 player, together with original task examples and generic replay. We keep these versions in separate checkpoint folders, so we can load and compare each one. A new version records a training experiment; its name alone does not establish that it survives longer or collects pellets more efficiently.

The first equation describes what we copy. Phi is our name for the complete collection of trainable LoRA and pointer-head parameters. “Parent” identifies the checkpoint we load. “Child” identifies the new stage we are about to train. The superscript zero means its starting values. Before the child learns anything new, those values equal the parent’s saved values. The same fixed Qwen base supports both.

The second equation describes the training data. D-train is the collection of requests used by the new stage. D-new contains its new labels, and D-replay contains examples we keep practicing from earlier training. The combination symbol means we put those collections together, keeping repeated requests. In the actual code, we concatenate the request lists; it does not silently remove duplicates.

A new stage also starts a fresh optimizer and learning-rate schedule. That differs from recovering an interrupted run. Recovery restores the progress of that same stage, including its optimizer and step position. Here, we are following completed parents into new training stages. Next we need to be precise about what a Pac-Man state tells the model, and which behaviors our teacher should demonstrate.

Presenter reference: This diagram is the lab’s pinned 4B curriculum and its completed native-v2/v3 task lineage. The base revision is 1001bb4d826a52d1f399e183466143f4da7b741b. Initial: 12,576 requests × two epochs. Dates: 1,425 new + 2,000 generic replay; Documents: 5,219 + 2,000; Skills: 6,000 hard-v1 train + 5,320 devtools-v1 train + 4,000 replay, each one epoch. The Skills concatenation uses the published training partitions/counts; it is not asserted byte-identical to the unavailable historical joint file. V2: 4,096 task + 2,000 generic replay. V3: 2,048 original v2 + 2,048 verified correction/recovery + 2,000 generic replay. Both task stages are one epoch and 762 updates at effective batch eight. These are request counts, not counts of complete games or unique labeled questions. Generic augmentation may add question-level examples. Pac-Man v1 remains a separate archived task experiment, not a parent of this native-v2 → native-v3 chain. V4 dataset work is separate and no trained-v4 or improved gameplay is claimed. Warm-start copies compatible LoRA/head weights; temperature fitting is separate. The ⊎ equation denotes multiset combination, matching concatenation with repeated requests retained; it is not a new optimization objective.

Sources: https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/training-stages.json

Initial recipe: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/experiments/q35-4b-s23.json

Warm-start: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/checkpoint.py

Training: https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/kev/train.py

V3: https://github.com/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_v3.ipynb

Animation: https://yxc20089.github.io/QPlusLearning/animations/kev-curriculum.html
