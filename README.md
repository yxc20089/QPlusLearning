# Q+Learning

Hands-on labs for learning how language models make decisions, how to train them, and how to evaluate their behavior.

| Lab | What you build | Session |
| --- | --- | --- |
| [Lab 1 — Train a Decision Model to Play Pac-Man](labs/lab-01-kev-pacman/README.md) | A Kev-4B player using LoRA and a pointer head | 90 minutes with a 30-minute fine-tuning block; complete slower full training before class |

[![Open Lab 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

Lab 1 has one student notebook with shared setup, TensorBoard, Drive restore and a final backup. Its current CP1 verifies a completed native-v3 checkpoint, imports the frozen balanced dataset, trains native-v4, then compares complete native games and supports ungraded interactive play. The earlier general training stages and native-v2 experiment remain optional prework. Separate [v3](labs/lab-01-kev-pacman/notebooks/pacman_kev_v3.ipynb) and [v4](labs/lab-01-kev-pacman/notebooks/pacman_kev_v4.ipynb) notebooks remain available. Instructors must supply the exact v3 checkpoint and dataset artifacts; the notebook does not include private Drive files.

The [live lab slides](https://docs.google.com/presentation/d/1_PfmbUEH38jMO_24S_AqtUUh-IIg2LyZYkEYKjZ5uXQ/edit) explain the architecture, LoRA, pointer head and teacher with equations, natural speaker notes and [interactive animations](https://yxc20089.github.io/QPlusLearning/). The deck is being revised into 13 slides; local PDF/PPTX exports retain the earlier edition until that revision is complete. The [companion lecture](https://docs.google.com/presentation/d/1sdnPkV6VyTW9tr6Xmtyyxns4UHlUGoTtTLj-vmxNhfo/edit) provides further explanations.

The lab targets Colab with an NVIDIA RTX PRO 6000 Blackwell GPU. Arrange access before class; that allocation is not guaranteed or assumed free. One completed optimized v4 learner run took 64.2 minutes for 762 updates with 26.94 GiB peak allocated GPU memory. Its completion check passed; gameplay improvement still requires evaluation. CPU checks do not establish performance on other GPU allocations; see [setup](labs/lab-01-kev-pacman/COLAB_SETUP.md).

Course code and content use the [MIT license](LICENSE), with third-party materials covered by their [original licenses](labs/lab-01-kev-pacman/vendor/README.md).
