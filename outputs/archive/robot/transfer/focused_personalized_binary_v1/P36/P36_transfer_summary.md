# P36 final frozen EEG-only Image-to-Robot transfer

Predictions use frozen Image pipelines. Robot ratings are descriptive post-inference comparisons only.

| Task     | Target   |   Median P(HIGH) | Verdict   |   Agreement |   Robot rating | Rating class   | Match   | Shift warnings   |
|:---------|:---------|-----------------:|:----------|------------:|---------------:|:---------------|:--------|:-----------------|
| PnP      | arousal  |         0.864774 | HIGH      |    0.71875  |              5 | HIGH           | True    |                  |
| SSAlone  | arousal  |         0.890246 | HIGH      |    0.681818 |              2 | LOW            | False   |                  |
| SSInt    | arousal  |         0.897503 | HIGH      |    0.782051 |              3 | LOW            | False   |                  |
| SSObs    | arousal  |         0.813656 | HIGH      |    0.518868 |              6 | HIGH           | True    |                  |
| Sisyphus | arousal  |         0.873259 | HIGH      |    0.682171 |              7 | HIGH           | True    |                  |
| St       | arousal  |         0.649853 | HIGH      |    0.308824 |              4 | HIGH           | True    |                  |
| PnP      | valence  |         0.503768 | HIGH      |    0.625    |              5 | HIGH           | True    |                  |
| SSAlone  | valence  |         0.626971 | HIGH      |    0.545455 |              4 | HIGH           | True    |                  |
| SSInt    | valence  |         0.658542 | HIGH      |    0.679487 |              5 | HIGH           | True    |                  |
| SSObs    | valence  |         0.759883 | HIGH      |    0.783019 |              6 | HIGH           | True    |                  |
| Sisyphus | valence  |         0.731952 | HIGH      |    0.767442 |              7 | HIGH           | True    |                  |
| St       | valence  |         0.567567 | HIGH      |    0.514706 |              5 | HIGH           | True    |                  |
