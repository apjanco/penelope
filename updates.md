1. focus on typology -- what is/is not SoC? what are the types? 
2. annotation agreement OR large LLM silver data -- can we find type examples?
3. Train a classifier. User problem -- identify SoC at corpus scale [focus here]

## Why CoT/Reasoning Models Don't Help

Your finding makes sense. Chain-of-thought reasoning models are trained to produce *logical, sequential reasoning* — which is almost the opposite of stream of consciousness. CoT is structured, goal-directed thinking. SoC is associative, fragmented, and often non-linear. The "chain of thought" in their name refers to inference-time reasoning scaffolding, not any special sensitivity to how *humans* think in unstructured ways. There's no representational advantage there for your task.

## Thinking Through the Core Problem

The first question is what your typology looks like. SoC exists on a spectrum and has been theorized differently across literary criticism, psychology, and linguistics. A classifier needs categories that are both theoretically grounded and empirically distinguishable. Some dimensions I'd expect in a useful typology:

**Narrative technique** — interior monologue (first-person, grammatically coherent, as in Woolf), free indirect discourse (third-person blending narrator and character, as in Austen or Flaubert), direct interior monologue (raw, unmediated, as in parts of Molly Bloom's soliloquy), sensory impression cataloguing (as in some Nouveau Roman work).

**Degree of fragmentation** — ranging from syntactically complete but associatively linked passages, through to heavily fragmented, unpunctuated, syntactically broken text.

**Cognitive register** — perception-dominant (sensory cataloguing), memory-dominant (temporal shifting), affect-dominant (emotional coloring), ruminative (recursive return to themes).

Before training anything, I'd recommend formalizing your typology with an annotation guide and measuring inter-annotator agreement. If humans can't reliably distinguish your categories, a model won't either, and that's a signal to revise the categories.

## Approach to Training

Here's how I'd structure the pipeline:

**1. Data and annotation first, model second.** The bottleneck for this kind of task is almost never the model architecture — it's the quality and quantity of labeled data. SoC passages are relatively rare in general corpora, and expert annotation is expensive. I'd prioritize building a well-annotated dataset of maybe 1,000–2,000 passages (even 500 can work for fine-tuning if categories are clean) drawn from known SoC authors (Joyce, Woolf, Faulkner, Richardson, Beckett, Lispector, etc.) and non-SoC controls.

**2. Fine-tune a pre-trained language model, not train from scratch.** A model like RoBERTa, DeBERTa, or a Longformer (if your passages are long) already encodes deep syntactic and stylistic features. Fine-tuning on your labeled data is the most efficient path. These models already "know" things like sentence fragmentation, unusual punctuation patterns, tense shifting, and pronominal ambiguity — they just need to learn which patterns map to your categories.

**3. Consider a two-stage classifier.** Stage one: binary — is this SoC or not? Stage two: given that it's SoC, which subtype? This decomposition is useful because the SoC-vs-not boundary is likely more learnable than fine-grained subtypes, and it lets you evaluate each stage independently.

**4. Feature engineering as a complement, not a replacement.** Linguistic features can serve as additional inputs or as diagnostic tools for error analysis. Things worth extracting: sentence length variance (SoC tends to have high variance or extremely long sentences), punctuation density and type distribution (semicolons, dashes, ellipses), pronoun-to-noun ratio, temporal marker frequency and consistency, syntactic parse tree depth and branching patterns, type-token ratio (lexical diversity). These can be concatenated with transformer embeddings in a hybrid model, or used standalone in an interpretable baseline (logistic regression, gradient-boosted trees) that helps you understand what the model is learning.

**5. Few-shot and prompt-based classification as a baseline.** Before fine-tuning anything, test a strong general LLM (like Claude or GPT-4) with a well-crafted prompt that includes your typology definitions and a few examples per category. This gives you a ceiling estimate for zero/few-shot performance and can also serve as a cheap way to generate silver labels for pre-training before human annotation.

## Practical Suggestions

**On passage length:** SoC features often only become apparent over several sentences. I'd use passages of at least 150–300 words. Too short and you lose the associative drift that distinguishes subtypes.

**On negative examples:** Include near-misses — first-person narration that *isn't* SoC, lyrical prose, confessional writing, diary entries. The classifier needs to learn that interiority alone isn't sufficient.

**On evaluation:** Use stratified cross-validation given your likely small dataset. Report per-class F1, not just accuracy, since your categories will almost certainly be imbalanced. A confusion matrix will tell you which subtypes the model conflates, which feeds back into refining the typology.

**On interpretability:** If this is for a research paper, reviewers will want to know *what* the model learned, not just that it works. Techniques like integrated gradients, SHAP over token embeddings, or even simpler attention visualization can show which textual features drive classification decisions — and that's publishable analysis in its own right.

## What I'd Recommend as a Starting Pipeline

The most productive path forward is probably: define your typology and annotation guidelines, then annotate a pilot set of around 100 passages and measure agreement, then use few-shot LLM classification to generate candidate labels on a larger set, then have humans correct those labels (faster than annotating from scratch), then fine-tune DeBERTa-base or Longformer on the corrected set, and finally run the linguistic feature analysis in parallel for interpretability and as a diagnostic baseline.

Want me to dig into any of these pieces — the typology design, the annotation protocol, the model architecture choice, or the feature engineering? I can also help draft the annotation guidelines or sketch out the training code if you have a sense of what your categories are.
