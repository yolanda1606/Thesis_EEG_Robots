# P36 final transfer extended descriptive report

All quantities are post-inference summaries of saved predictions. Robot ratings are retained on their original 1–7 scale and used only for descriptive HIGH/LOW comparison; P(HIGH) is not mapped to that scale.

## Task-level consensus and agreement

| Task     | Robot condition   | Target   |   Median P(HIGH) |   Mean P(HIGH) |   Consensus windows ≥0.5 | Verdict   |   Mean range |   Median range |   Mean pairwise |ΔP| |   Unanimous LOW |   Unanimous HIGH |   Unanimous |   Majority HIGH |   Self-report (1–7) | Rating class   | Verdict match   |
|:---------|:------------------|:---------|-----------------:|---------------:|-------------------------:|:----------|-------------:|---------------:|---------------------:|----------------:|-----------------:|------------:|----------------:|--------------------:|:---------------|:----------------|
| PnP      | FAULTY            | arousal  |         0.833253 |       0.810735 |                 1        | HIGH      |     0.312515 |       0.313169 |            0.208344  |       0         |         0.71875  |    0.71875  |        0.90625  |                   5 | HIGH           | True            |
| SSAlone  | N/A               | arousal  |         0.881937 |       0.849085 |                 1        | HIGH      |     0.298467 |       0.314674 |            0.198978  |       0         |         0.681818 |    0.681818 |        0.909091 |                   2 | LOW            | False           |
| SSInt    | N/A               | arousal  |         0.889343 |       0.839274 |                 0.987179 | HIGH      |     0.308016 |       0.315906 |            0.205344  |       0.0384615 |         0.74359  |    0.782051 |        0.935897 |                   3 | LOW            | False           |
| SSObs    | SLOW              | arousal  |         0.766123 |       0.744567 |                 0.962264 | HIGH      |     0.298135 |       0.318103 |            0.198757  |       0.0471698 |         0.471698 |    0.518868 |        0.811321 |                   6 | HIGH           | True            |
| Sisyphus | N/A               | arousal  |         0.884626 |       0.791363 |                 0.945736 | HIGH      |     0.275138 |       0.284742 |            0.183425  |       0.0620155 |         0.620155 |    0.682171 |        0.821705 |                   7 | HIGH           | True            |
| St       | FAST              | arousal  |         0.649853 |       0.662658 |                 0.897059 | HIGH      |     0.284759 |       0.283404 |            0.189839  |       0.117647  |         0.191176 |    0.308824 |        0.691176 |                   4 | HIGH           | True            |
| PnP      | FAULTY            | valence  |         0.497968 |       0.551737 |                 0.5      | LOW       |     0.163671 |       0.153632 |            0.109114  |       0.3125    |         0.3125   |    0.625    |        0.46875  |                   5 | HIGH           | False           |
| SSAlone  | N/A               | valence  |         0.618376 |       0.575917 |                 0.590909 | HIGH      |     0.151039 |       0.117924 |            0.100693  |       0.136364  |         0.409091 |    0.545455 |        0.590909 |                   4 | HIGH           | True            |
| SSInt    | N/A               | valence  |         0.658542 |       0.634335 |                 0.679487 | HIGH      |     0.15902  |       0.154498 |            0.106013  |       0.217949  |         0.461538 |    0.679487 |        0.641026 |                   5 | HIGH           | True            |
| SSObs    | SLOW              | valence  |         0.759883 |       0.71345  |                 0.867925 | HIGH      |     0.182297 |       0.182587 |            0.121531  |       0.113208  |         0.669811 |    0.783019 |        0.839623 |                   6 | HIGH           | True            |
| Sisyphus | N/A               | valence  |         0.731952 |       0.686517 |                 0.821705 | HIGH      |     0.149834 |       0.143708 |            0.0998891 |       0.124031  |         0.643411 |    0.767442 |        0.821705 |                   7 | HIGH           | True            |
| St       | FAST              | valence  |         0.566497 |       0.583808 |                 0.705882 | HIGH      |     0.142372 |       0.150934 |            0.0949147 |       0.132353  |         0.382353 |    0.514706 |        0.617647 |                   5 | HIGH           | True            |

## Historical comparison

No historical comparison directory was supplied.
