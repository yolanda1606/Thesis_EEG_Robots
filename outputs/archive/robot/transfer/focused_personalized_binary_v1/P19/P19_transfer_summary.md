# P19 final frozen EEG-only Image-to-Robot transfer

Predictions use frozen Image pipelines. Robot ratings are descriptive post-inference comparisons only.

| Task     | Target   |   Median P(HIGH) | Verdict   |   Agreement |   Robot rating | Rating class   | Match   | Shift warnings   |
|:---------|:---------|-----------------:|:----------|------------:|---------------:|:---------------|:--------|:-----------------|
| PnP      | arousal  |        0.141339  | LOW       |    0.97619  |              5 | HIGH           | False   |                  |
| SSAlone  | arousal  |        0.0144035 | LOW       |    0.962963 |              3 | LOW            | True    |                  |
| SSInt    | arousal  |        0.0284422 | LOW       |    0.961039 |              6 | HIGH           | False   |                  |
| SSObs    | arousal  |        0.289119  | LOW       |    0.813559 |              5 | HIGH           | False   |                  |
| Sisyphus | arousal  |        0.0295651 | LOW       |    0.956044 |              5 | HIGH           | False   |                  |
| St       | arousal  |        0.11277   | LOW       |    0.941176 |              4 | HIGH           | False   |                  |
| PnP      | valence  |        0.642089  | HIGH      |    0.714286 |              5 | HIGH           | True    |                  |
| SSAlone  | valence  |        0.237503  | LOW       |    0.666667 |              5 | HIGH           | False   |                  |
| SSInt    | valence  |        0.291313  | LOW       |    0.74026  |              6 | HIGH           | False   |                  |
| SSObs    | valence  |        0.554838  | HIGH      |    0.728814 |              5 | HIGH           | True    |                  |
| Sisyphus | valence  |        0.359448  | LOW       |    0.78022  |              4 | HIGH           | False   |                  |
| St       | valence  |        0.418953  | LOW       |    0.661765 |              4 | HIGH           | False   |                  |
