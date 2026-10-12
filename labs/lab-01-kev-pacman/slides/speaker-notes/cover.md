Welcome to Lab 1. We are going to train Kev to play Pac-Man, using the classic game with four different ghost controllers.

Before we begin, open these two links. The GitHub repository contains the lab code, instructions and supporting resources. The Colab link opens our main student notebook, with shared setup, the training workflow, full-game evaluation, interactive play and the final Drive backup in one place. Keep both tabs open so you can move between the explanation and the code.

We will first see how Kev turns a board and a set of legal moves into action probabilities. Then we will look at the LoRA adapters and pointer head that learn from an offline teacher. Finally, we will let the trained model make its own choices in complete games and examine whether it avoids ghosts and keeps collecting pellets.

Let's start with the architecture.

Resources:
https://github.com/yxc20089/QPlusLearning
https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb
