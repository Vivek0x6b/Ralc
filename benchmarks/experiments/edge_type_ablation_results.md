# Edge-type ablation

- generated: 2026-10-09T09:08:06.393300+00:00
- metric columns: relationship complete-story / recall, lookup complete-story / recall

## v1 (relationship 8, lookup 6; full RALC avg pool 71.9)

Matched seed_k (avg pool): full_ralc 10 (71.9), remove_TEMPORALLY_FOLLOWS 29 (72.6), remove_ANSWERS 10 (71.9), remove_MENTIONS 17 (74.1), only_TEMPORALLY_FOLLOWS 17 (74.1)

### Budget 1000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.88 | 0.94 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.75 | 0.90 | 1.00 | 1.00 |
| remove_ANSWERS | 0.75 | 0.90 | 1.00 | 1.00 |
| remove_MENTIONS | 0.88 | 0.96 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.88 | 0.96 | 1.00 | 1.00 |

### Budget 2000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 1.00 | 1.00 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.88 | 0.96 | 1.00 | 1.00 |
| remove_ANSWERS | 1.00 | 1.00 | 1.00 | 1.00 |
| remove_MENTIONS | 1.00 | 1.00 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 4000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 1.00 | 1.00 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 1.00 | 1.00 | 1.00 | 1.00 |
| remove_ANSWERS | 1.00 | 1.00 | 1.00 | 1.00 |
| remove_MENTIONS | 1.00 | 1.00 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 1.00 | 1.00 | 1.00 | 1.00 |

Mean relationship complete-story over budgets: full_ralc 0.96, remove_TEMPORALLY_FOLLOWS 0.88, remove_ANSWERS 0.92, remove_MENTIONS 0.96, only_TEMPORALLY_FOLLOWS 0.96

Biggest drop from removing one edge type: remove_TEMPORALLY_FOLLOWS (-0.08 vs full RALC 0.96).

## v2 (relationship 11, lookup 6; full RALC avg pool 60.9)

Matched seed_k (avg pool): full_ralc 10 (60.9), remove_TEMPORALLY_FOLLOWS 24 (60.4), remove_ANSWERS 10 (60.9), remove_MENTIONS 14 (59.5), only_TEMPORALLY_FOLLOWS 14 (59.5)

### Budget 250

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.45 | 0.62 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.45 | 0.67 | 1.00 | 1.00 |
| remove_ANSWERS | 0.45 | 0.62 | 1.00 | 1.00 |
| remove_MENTIONS | 0.45 | 0.62 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.45 | 0.62 | 1.00 | 1.00 |

### Budget 500

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.73 | 0.86 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.64 | 0.83 | 1.00 | 1.00 |
| remove_ANSWERS | 0.64 | 0.83 | 1.00 | 1.00 |
| remove_MENTIONS | 0.64 | 0.83 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.64 | 0.83 | 1.00 | 1.00 |

### Budget 1000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.73 | 0.91 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.82 | 0.92 | 1.00 | 1.00 |
| remove_ANSWERS | 0.64 | 0.86 | 1.00 | 1.00 |
| remove_MENTIONS | 0.64 | 0.86 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.64 | 0.86 | 1.00 | 1.00 |

### Budget 2000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.82 | 0.93 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.91 | 0.98 | 1.00 | 1.00 |
| remove_ANSWERS | 0.82 | 0.93 | 1.00 | 1.00 |
| remove_MENTIONS | 0.73 | 0.91 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.73 | 0.89 | 1.00 | 1.00 |

### Budget 4000

| method | rel complete | rel recall | lookup complete | lookup recall |
| --- | --- | --- | --- | --- |
| full_ralc | 0.82 | 0.93 | 1.00 | 1.00 |
| remove_TEMPORALLY_FOLLOWS | 0.91 | 0.98 | 1.00 | 1.00 |
| remove_ANSWERS | 0.82 | 0.93 | 1.00 | 1.00 |
| remove_MENTIONS | 0.82 | 0.93 | 1.00 | 1.00 |
| only_TEMPORALLY_FOLLOWS | 0.82 | 0.93 | 1.00 | 1.00 |

Mean relationship complete-story over budgets: full_ralc 0.71, remove_TEMPORALLY_FOLLOWS 0.75, remove_ANSWERS 0.67, remove_MENTIONS 0.65, only_TEMPORALLY_FOLLOWS 0.65

Biggest drop from removing one edge type: remove_MENTIONS (-0.05 vs full RALC 0.71).
