GRPO underperforms SFT on this evaluation.

Binary detection — the core task — is worse across every metric: SoC F1 drops from 45.4% → 40.7%, accuracy from 75.9% → 73.0%. GRPO is both less precise and less sensitive at detecting SoC passages.

Type classification is poor for both, but the story is mixed:

On the two classes that actually have meaningful support, SFT is considerably better: direct_interior_monologue (34% → 15%) and indirect_interior_monologue (31% → 24%).
GRPO's tiny macro F1 gain (+0.5 points) comes entirely from simulation_state_of_mind (3 examples) and free_association (1 example) — not reliable signal.
19 of 23 classes score 0% for both models, so type classification is largely unsolved either way.
Likely explanations for GRPO regressing:

Likely explanations for GRPO regressing:

The reward signal wasn't closely aligned with precision/recall on these specific metrics — GRPO may have learned to produce well-formatted JSON without improving detection quality.
checkpoint-5291 may not be the best GRPO checkpoint — worth checking the training reward curve to see if an earlier checkpoint performed better.
GRPO can hurt supervised signal if the reward is too sparse or noisy, effectively unlearning some of what SFT learned.
The headline is: SFT is currently the stronger model for deployment, and the GRPO reward design likely needs revisiting.