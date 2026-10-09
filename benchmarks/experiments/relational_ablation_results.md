# Relation-aware selection ablation

- generated: 2026-10-09T08:57:08.878648+00:00

## v1 (relationship 8, lookup 6; gemma cache misses 2; full RALC avg pool 71.9, relations_off_matched seed_k 72)

### Budget 1000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.00 | 998 | 0/6 |
| recent | lookup | 0.00 | 0.00 | 998 | 0/6 |
| vector | relationship | 0.75 | 0.90 | 990 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 994 | 0/0 |
| graph | relationship | 1.00 | 1.00 | 985 | 2/0 |
| graph | lookup | 1.00 | 1.00 | 973 | 0/0 |
| ralc_heuristic | relationship | 0.88 | 0.94 | 987 | 1/0 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 969 | 0/0 |
| relations_off | relationship | 0.75 | 0.90 | 443 | 0/0 |
| relations_off | lookup | 1.00 | 1.00 | 477 | 0/0 |
| relations_off_matched | relationship | 0.75 | 0.85 | 987 | 0/0 |
| relations_off_matched | lookup | 1.00 | 1.00 | 984 | 0/0 |
| redundancy_off | relationship | 1.00 | 1.00 | 985 | 2/0 |
| redundancy_off | lookup | 1.00 | 1.00 | 978 | 0/0 |
| relation_aware | relationship | 0.88 | 0.94 | 985 | 1/0 |
| relation_aware | lookup | 1.00 | 1.00 | 969 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21051 | 2/0 |
| full | lookup | 1.00 | 1.00 | 21051 | 0/0 |
| ralc_gemma | relationship | 0.88 | 0.94 | 986 | 1/0 |
| ralc_gemma | lookup | 1.00 | 1.00 | 980 | 0/0 |
| ralc_gemma_relation_aware | relationship | 1.00 | 1.00 | 986 | 2/0 |
| ralc_gemma_relation_aware | lookup | 1.00 | 1.00 | 977 | 0/0 |

### Budget 2000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.00 | 1983 | 0/7 |
| recent | lookup | 0.00 | 0.00 | 1983 | 0/6 |
| vector | relationship | 0.88 | 0.96 | 1990 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 1989 | 0/0 |
| graph | relationship | 1.00 | 1.00 | 1980 | 1/0 |
| graph | lookup | 1.00 | 1.00 | 1990 | 0/0 |
| ralc_heuristic | relationship | 1.00 | 1.00 | 1957 | 1/0 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 1854 | 0/0 |
| relations_off | relationship | 0.75 | 0.90 | 443 | 0/1 |
| relations_off | lookup | 1.00 | 1.00 | 477 | 0/0 |
| relations_off_matched | relationship | 0.75 | 0.90 | 1962 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 1708 | 0/0 |
| redundancy_off | relationship | 1.00 | 1.00 | 1989 | 1/0 |
| redundancy_off | lookup | 1.00 | 1.00 | 1985 | 0/0 |
| relation_aware | relationship | 1.00 | 1.00 | 1957 | 1/0 |
| relation_aware | lookup | 1.00 | 1.00 | 1854 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21051 | 1/0 |
| full | lookup | 1.00 | 1.00 | 21051 | 0/0 |
| ralc_gemma | relationship | 1.00 | 1.00 | 1985 | 1/0 |
| ralc_gemma | lookup | 1.00 | 1.00 | 1984 | 0/0 |
| ralc_gemma_relation_aware | relationship | 1.00 | 1.00 | 1988 | 1/0 |
| ralc_gemma_relation_aware | lookup | 1.00 | 1.00 | 1990 | 0/0 |

### Budget 4000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.00 | 3993 | 0/8 |
| recent | lookup | 0.00 | 0.00 | 3993 | 0/6 |
| vector | relationship | 1.00 | 1.00 | 3985 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 3990 | 0/0 |
| graph | relationship | 1.00 | 1.00 | 3975 | 0/0 |
| graph | lookup | 1.00 | 1.00 | 3966 | 0/0 |
| ralc_heuristic | relationship | 1.00 | 1.00 | 2670 | 0/0 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 2319 | 0/0 |
| relations_off | relationship | 0.75 | 0.90 | 443 | 0/2 |
| relations_off | lookup | 1.00 | 1.00 | 477 | 0/0 |
| relations_off_matched | relationship | 0.88 | 0.94 | 2298 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 1798 | 0/0 |
| redundancy_off | relationship | 1.00 | 1.00 | 3981 | 0/0 |
| redundancy_off | lookup | 1.00 | 1.00 | 3974 | 0/0 |
| relation_aware | relationship | 1.00 | 1.00 | 2670 | 0/0 |
| relation_aware | lookup | 1.00 | 1.00 | 2319 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21051 | 0/0 |
| full | lookup | 1.00 | 1.00 | 21051 | 0/0 |
| ralc_gemma | relationship | 1.00 | 1.00 | 3966 | 0/0 |
| ralc_gemma | lookup | 1.00 | 1.00 | 3238 | 0/0 |
| ralc_gemma_relation_aware | relationship | 1.00 | 1.00 | 3974 | 0/0 |
| ralc_gemma_relation_aware | lookup | 1.00 | 1.00 | 3272 | 0/0 |

## v2 (relationship 11, lookup 6; gemma cache misses None; full RALC avg pool 60.9, relations_off_matched seed_k 61)

### Budget 250

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.00 | 247 | 0/5 |
| recent | lookup | 0.17 | 0.17 | 247 | 0/5 |
| vector | relationship | 0.45 | 0.67 | 235 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 228 | 0/0 |
| graph | relationship | 0.45 | 0.67 | 232 | 0/0 |
| graph | lookup | 1.00 | 1.00 | 228 | 0/0 |
| ralc_heuristic | relationship | 0.45 | 0.62 | 236 | 0/0 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 230 | 0/0 |
| relations_off | relationship | 0.27 | 0.52 | 227 | 1/3 |
| relations_off | lookup | 1.00 | 1.00 | 220 | 0/0 |
| relations_off_matched | relationship | 0.18 | 0.52 | 234 | 1/4 |
| relations_off_matched | lookup | 1.00 | 1.00 | 232 | 0/0 |
| redundancy_off | relationship | 0.45 | 0.65 | 234 | 0/0 |
| redundancy_off | lookup | 1.00 | 1.00 | 225 | 0/0 |
| relation_aware | relationship | 0.45 | 0.62 | 236 | 0/0 |
| relation_aware | lookup | 1.00 | 1.00 | 230 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21436 | 6/0 |
| full | lookup | 1.00 | 1.00 | 21436 | 0/0 |

### Budget 500

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.11 | 477 | 0/7 |
| recent | lookup | 0.33 | 0.33 | 477 | 0/4 |
| vector | relationship | 0.64 | 0.79 | 492 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 491 | 0/0 |
| graph | relationship | 0.64 | 0.81 | 485 | 0/0 |
| graph | lookup | 1.00 | 1.00 | 486 | 0/0 |
| ralc_heuristic | relationship | 0.73 | 0.86 | 490 | 1/0 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 484 | 0/0 |
| relations_off | relationship | 0.64 | 0.80 | 443 | 0/0 |
| relations_off | lookup | 1.00 | 1.00 | 422 | 0/0 |
| relations_off_matched | relationship | 0.55 | 0.70 | 487 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 486 | 0/0 |
| redundancy_off | relationship | 0.73 | 0.89 | 486 | 1/0 |
| redundancy_off | lookup | 1.00 | 1.00 | 486 | 0/0 |
| relation_aware | relationship | 0.73 | 0.89 | 489 | 1/0 |
| relation_aware | lookup | 1.00 | 1.00 | 484 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21436 | 4/0 |
| full | lookup | 1.00 | 1.00 | 21436 | 0/0 |

### Budget 1000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.11 | 987 | 0/8 |
| recent | lookup | 0.33 | 0.33 | 987 | 0/4 |
| vector | relationship | 0.73 | 0.90 | 990 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 989 | 0/0 |
| graph | relationship | 0.73 | 0.89 | 989 | 1/1 |
| graph | lookup | 1.00 | 1.00 | 989 | 0/0 |
| ralc_heuristic | relationship | 0.73 | 0.91 | 982 | 1/1 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 983 | 0/0 |
| relations_off | relationship | 0.64 | 0.83 | 484 | 0/1 |
| relations_off | lookup | 1.00 | 1.00 | 440 | 0/0 |
| relations_off_matched | relationship | 0.64 | 0.80 | 987 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 976 | 0/0 |
| redundancy_off | relationship | 0.73 | 0.91 | 988 | 1/1 |
| redundancy_off | lookup | 1.00 | 1.00 | 988 | 0/0 |
| relation_aware | relationship | 0.73 | 0.91 | 978 | 1/1 |
| relation_aware | lookup | 1.00 | 1.00 | 986 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21436 | 3/0 |
| full | lookup | 1.00 | 1.00 | 21436 | 0/0 |

### Budget 2000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.11 | 2000 | 0/9 |
| recent | lookup | 0.33 | 0.33 | 2000 | 0/4 |
| vector | relationship | 0.82 | 0.92 | 1991 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 1989 | 0/0 |
| graph | relationship | 0.82 | 0.93 | 1982 | 2/2 |
| graph | lookup | 1.00 | 1.00 | 1967 | 0/0 |
| ralc_heuristic | relationship | 0.82 | 0.93 | 1924 | 2/2 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 1747 | 0/0 |
| relations_off | relationship | 0.64 | 0.83 | 484 | 0/2 |
| relations_off | lookup | 1.00 | 1.00 | 440 | 0/0 |
| relations_off_matched | relationship | 0.73 | 0.88 | 1958 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 1687 | 0/0 |
| redundancy_off | relationship | 0.73 | 0.91 | 1978 | 1/2 |
| redundancy_off | lookup | 1.00 | 1.00 | 1991 | 0/0 |
| relation_aware | relationship | 0.82 | 0.93 | 1929 | 2/2 |
| relation_aware | lookup | 1.00 | 1.00 | 1753 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21436 | 2/0 |
| full | lookup | 1.00 | 1.00 | 21436 | 0/0 |

### Budget 4000

| method | type | complete-story | mean recall | mean tokens | win/loss vs vector |
| --- | --- | --- | --- | --- | --- |
| recent | relationship | 0.00 | 0.11 | 3978 | 0/9 |
| recent | lookup | 0.33 | 0.33 | 3978 | 0/4 |
| vector | relationship | 0.82 | 0.95 | 3993 | 0/0 |
| vector | lookup | 1.00 | 1.00 | 3990 | 0/0 |
| graph | relationship | 0.82 | 0.93 | 3708 | 2/2 |
| graph | lookup | 1.00 | 1.00 | 3805 | 0/0 |
| ralc_heuristic | relationship | 0.82 | 0.93 | 2369 | 2/2 |
| ralc_heuristic | lookup | 1.00 | 1.00 | 2204 | 0/0 |
| relations_off | relationship | 0.64 | 0.83 | 484 | 0/2 |
| relations_off | lookup | 1.00 | 1.00 | 440 | 0/0 |
| relations_off_matched | relationship | 0.73 | 0.88 | 2123 | 0/1 |
| relations_off_matched | lookup | 1.00 | 1.00 | 1752 | 0/0 |
| redundancy_off | relationship | 0.82 | 0.93 | 3714 | 2/2 |
| redundancy_off | lookup | 1.00 | 1.00 | 3820 | 0/0 |
| relation_aware | relationship | 0.82 | 0.93 | 2377 | 2/2 |
| relation_aware | lookup | 1.00 | 1.00 | 2210 | 0/0 |
| full | relationship | 1.00 | 1.00 | 21436 | 2/0 |
| full | lookup | 1.00 | 1.00 | 21436 | 0/0 |
