"""Style-only reference notes derived from the CAMS paper (Garg et al., 2022,
LREC), used the same way the sibling project's cams_examples.py uses it: as a
concreteness/structure anchor for how real people phrase distress on Reddit,
NOT as content to copy or as a stigma-measurement source.

Why CAMS is used narrowly here:
  - CAMS annotates *causal category* of depression/suicide-risk posts (no
    reason, bias/abuse, jobs/careers, medication, relationship, alienation).
    It is not an internalized-stigma (ISMI) dataset; "alienation" there means
    meaninglessness/isolation, not shame-driven self-stigma specifically.
  - CAMS is sourced from r/depression and the SDCNL dataset (itself drawn from
    r/depression and r/SuicideWatch). Several published examples are
    crisis-adjacent (e.g. explicit suicidal ideation, hospitalization). Those
    are intentionally excluded below -- this file only paraphrases STRUCTURE
    from the paper's non-crisis examples, never verbatim text, and never the
    crisis-adjacent ones.
  - The paper itself restricts the raw corpus to selective, IRB-gated sharing
    (Section 4.4). No CAMS text is reproduced verbatim in this repo; only a
    generalized description of sentence-level structure is kept.

Source: Garg, M. et al. (2022). "CAMS: An Annotated Corpus for Causal Analysis
of Mental Health Issues in Social Media Posts." LREC 2022, pp. 6387-6396.
"""

# Paraphrased structural patterns observed in CAMS's non-crisis examples
# (Table 2: Alienation, Jobs and Careers, Relationships, Bias or Abuse
# categories). These describe SENTENCE STRUCTURE ONLY, not any specific
# person's content, and are meant to be blended into persona writing styles.
STRUCTURAL_PATTERNS = [
    "short clauses run together with '..' instead of a period, piling up "
    "small complaints before landing on one blunt self-label",
    "a thought trails off mid-sentence with '...' when the person seems to "
    "give up on finishing it",
    "a rhetorical, deflating question dropped in in the middle of a "
    "complaint (e.g. a quiet 'why bother' aside)",
    "matter-of-fact narration of an event that quietly slides into "
    "self-doubt or rationalization by the end of the same message",
    "dry, understated self-deprecation delivered almost as an aside rather "
    "than a direct statement of distress",
]
