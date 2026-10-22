# Style examples in the Patient prompt

Prompt 1 shows the Patient three short public Reddit posts from its diagnosis group (depression, anxiety, or PTSD)
so that its posts sound like posts on a support forum; the prompt tells it not to copy their wording. They are the
only real-person text in the pipeline. They were taken from the **Reddit Mental Health Dataset** (Low et al., 2020,
*JMIR* 22(10):e22635; Zenodo doi:10.5281/zenodo.3941387), released under the ODC Public Domain Dedication and
License 1.0, through the Hugging Face copy `kamruzzaman-asif/reddit-mental-health-classification` (same 28
subreddits), after filtering out clinical, third-person, advice-giving, and crisis posts.

We do not redistribute them. `outputs_v2/personas.json` ships with `few_shot_examples` empty, and
`outputs_v2/style_examples_manifest.json` records the length and SHA-256 of each excerpt. To rerun the experiment,
download from https://zenodo.org/records/3941387 the twelve files `{depression,anxiety,ptsd}_{2018,2019,pre,post}_features_tfidf_256.csv`
(about 520 MB) and run

    python3 fill_style_examples.py --data /path/to/{depression,anxiety,ptsd}_*_features_tfidf_256.csv

which locates the nine excerpts by hash (whitespace collapsed) and fills them in; this was checked to reproduce the
personas file the reported runs used exactly. The Hugging Face copy works too (columns `label`, `text`). `scripts/run_cell.sh` refuses to start while they are
empty. The released run records do not contain them.
