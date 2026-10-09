# Relational ablation with corrected complete-story metric and tags

- generated: 2026-10-09T10:09:10.924270+00:00
- OLD = full gold selected, tag from gold count. NEW = required subset selected, tag from required count.

## v1 (old tags: rel 8 / lookup 6; new tags: rel 6 / lookup 8)

### Budget 1000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.00 | 0.00 |
| vector | 0.75 | 0.67 | 1.00 | 1.00 |
| graph | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_heuristic | 0.88 | 0.83 | 1.00 | 1.00 |
| relations_off | 0.75 | 0.67 | 1.00 | 1.00 |
| relations_off_matched | 0.75 | 0.67 | 1.00 | 1.00 |
| redundancy_off | 1.00 | 1.00 | 1.00 | 1.00 |
| relation_aware | 0.88 | 0.83 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_gemma | 0.88 | 0.83 | 1.00 | 1.00 |
| ralc_gemma_relation_aware | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 2000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.00 | 0.00 |
| vector | 0.88 | 0.83 | 1.00 | 1.00 |
| graph | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_heuristic | 1.00 | 1.00 | 1.00 | 1.00 |
| relations_off | 0.75 | 0.67 | 1.00 | 1.00 |
| relations_off_matched | 0.75 | 0.67 | 1.00 | 1.00 |
| redundancy_off | 1.00 | 1.00 | 1.00 | 1.00 |
| relation_aware | 1.00 | 1.00 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_gemma | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_gemma_relation_aware | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 4000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.00 | 0.00 |
| vector | 1.00 | 1.00 | 1.00 | 1.00 |
| graph | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_heuristic | 1.00 | 1.00 | 1.00 | 1.00 |
| relations_off | 0.75 | 0.67 | 1.00 | 1.00 |
| relations_off_matched | 0.88 | 0.83 | 1.00 | 1.00 |
| redundancy_off | 1.00 | 1.00 | 1.00 | 1.00 |
| relation_aware | 1.00 | 1.00 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_gemma | 1.00 | 1.00 | 1.00 | 1.00 |
| ralc_gemma_relation_aware | 1.00 | 1.00 | 1.00 | 1.00 |

## v2 (old tags: rel 11 / lookup 6; new tags: rel 9 / lookup 8)

### Budget 250 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.17 | 0.12 |
| vector | 0.45 | 0.33 | 1.00 | 1.00 |
| graph | 0.45 | 0.33 | 1.00 | 1.00 |
| ralc_heuristic | 0.45 | 0.33 | 1.00 | 1.00 |
| relations_off | 0.27 | 0.11 | 1.00 | 1.00 |
| relations_off_matched | 0.18 | 0.11 | 1.00 | 0.88 |
| redundancy_off | 0.45 | 0.33 | 1.00 | 1.00 |
| relation_aware | 0.45 | 0.33 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 500 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.33 | 0.25 |
| vector | 0.64 | 0.56 | 1.00 | 1.00 |
| graph | 0.64 | 0.56 | 1.00 | 1.00 |
| ralc_heuristic | 0.73 | 0.67 | 1.00 | 1.00 |
| relations_off | 0.64 | 0.56 | 1.00 | 1.00 |
| relations_off_matched | 0.55 | 0.44 | 1.00 | 1.00 |
| redundancy_off | 0.73 | 0.67 | 1.00 | 1.00 |
| relation_aware | 0.73 | 0.67 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 1000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.33 | 0.25 |
| vector | 0.73 | 0.67 | 1.00 | 1.00 |
| graph | 0.73 | 0.67 | 1.00 | 1.00 |
| ralc_heuristic | 0.73 | 0.67 | 1.00 | 1.00 |
| relations_off | 0.64 | 0.56 | 1.00 | 1.00 |
| relations_off_matched | 0.64 | 0.56 | 1.00 | 1.00 |
| redundancy_off | 0.73 | 0.67 | 1.00 | 1.00 |
| relation_aware | 0.73 | 0.67 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 2000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.33 | 0.25 |
| vector | 0.82 | 0.78 | 1.00 | 1.00 |
| graph | 0.82 | 0.78 | 1.00 | 1.00 |
| ralc_heuristic | 0.82 | 0.78 | 1.00 | 1.00 |
| relations_off | 0.64 | 0.56 | 1.00 | 1.00 |
| relations_off_matched | 0.73 | 0.67 | 1.00 | 1.00 |
| redundancy_off | 0.73 | 0.67 | 1.00 | 1.00 |
| relation_aware | 0.82 | 0.78 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |

### Budget 4000 (complete-story: old -> new)

| method | relationship old | relationship new | lookup old | lookup new |
| --- | --- | --- | --- | --- |
| recent | 0.00 | 0.00 | 0.33 | 0.25 |
| vector | 0.82 | 0.78 | 1.00 | 1.00 |
| graph | 0.82 | 0.78 | 1.00 | 1.00 |
| ralc_heuristic | 0.82 | 0.78 | 1.00 | 1.00 |
| relations_off | 0.64 | 0.56 | 1.00 | 1.00 |
| relations_off_matched | 0.73 | 0.67 | 1.00 | 1.00 |
| redundancy_off | 0.82 | 0.78 | 1.00 | 1.00 |
| relation_aware | 0.82 | 0.78 | 1.00 | 1.00 |
| full | 1.00 | 1.00 | 1.00 | 1.00 |
