# Results (full)

Offline proxy metrics on a chronological MovieLens split. Smoke results are not the full training schedule. Do not read these numbers as user satisfaction or as a result about a live LLM.

Selected alpha: `1.0`

| Model | HR@10 | NDCG@10 | MRR | Coverage@10 | Diversity@10 | Novelty@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| popularity_bayesian | 0.0405 | 0.0166 | 0.0165 | 0.0321 | 0.7399 | 2.3563 |
| popularity_positive_count | 0.0682 | 0.0343 | 0.0334 | 0.0565 | 0.6853 | 1.2981 |
| matrix_factorization | 0.0586 | 0.0282 | 0.0259 | 0.0791 | 0.6564 | 1.5887 |
| content | 0.0213 | 0.0105 | 0.0130 | 0.2669 | 0.1334 | 4.5930 |
| hybrid | 0.0586 | 0.0282 | 0.0259 | 0.0791 | 0.6564 | 1.5887 |

## Explicit-rating metrics

- Validation RMSE: 0.9030528069387214
- Validation MAE: 0.7703120112419128
- Test RMSE: 0.8694328931102738
- Test MAE: 0.738447368144989

RMSE and MAE are secondary. The research comparison uses the ranking metrics above.

## LLM rerank on the shared sample

The LLM can only reorder the hybrid candidate set, so it cannot recover a relevant movie that the hybrid model did not nominate. Mock-client numbers are not measurements of gemma4-31b-it. A bootstrap interval is not a p-value. A small public dataset cannot establish online satisfaction.

- Result type: `full`
- Mock client: `False`
- Users: 100
- Mean paired NDCG@10 difference (LLM - hybrid): 0.0109
- 95% bootstrap interval: [-0.0132, 0.0348]
- Interval excludes zero: `False`
- Reading: The 95% bootstrap interval includes zero. This sample does not show a stable average difference. No p-value is reported.
- Top-1 change rate: 0.9000
- Mean absolute rank displacement: 5.8240
- Invalid response rate: 0.0400
- Fallback rate: 0.0000
- Cache hit rate: 0.0015873015873015873
- Average request latency seconds: 7.647075119980907
- Average prompt tokens: 2327.36
- Uncached client calls: 629
