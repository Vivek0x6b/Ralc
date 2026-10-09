# Topic-linking report

- generated: 2026-10-09T12:29:03.506741+00:00
- primary threshold: 0.6; sensitivity: [0.5, 0.6, 0.7]

## v1 (relationship 6, lookup 8; heuristic fallbacks 10)

UPDATES edge counts: entity_only 18, topic_0.50 23, topic_0.60 23, topic_0.70 23

Decision pairs:
- database ['m10', 'm100']: topic_similarity=0.23 topics=['database choice', 'moving the main store to Postgres']
    entity_only link: None
    topic@0.60 link: None
    clears threshold: {'0.50': False, '0.60': False, '0.70': False}
- hosting ['m33', 'm137']: topic_similarity=0.297 topics=['deployment platform', 'migrating app hosting to AWS']
    entity_only link: {'reason': 'entity', 'shared_entity': 'heroku', 'similarity': None}
    topic@0.60 link: {'reason': 'entity', 'shared_entity': 'heroku', 'similarity': None}
    clears threshold: {'0.50': False, '0.60': False, '0.70': False}

Eight decision-pair messages (is_decision, topic):
- m10: is_decision=True topic='database choice'
- m100: is_decision=True topic='moving the main store to Postgres'
- m33: is_decision=True topic='deployment platform'
- m137: is_decision=True topic='migrating app hosting to AWS'

Topic-created UPDATES edges at 0.60 (for false-link review): 5
- ['m60', 'm20'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m60] The window plants are dying of neglect, so Mei and I are starting a ve
    [m20] The window plants are dying of neglect, so Will and I are starting a v
- ['m99', 'm60'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m99] The window plants are dying of neglect, so Noah and I are starting a v
    [m60] The window plants are dying of neglect, so Mei and I are starting a ve
- ['m141', 'm99'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m141] The window plants are dying of neglect, so Dana and I are starting a v
    [m99] The window plants are dying of neglect, so Noah and I are starting a v
- ['m177', 'm141'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m177] The window plants are dying of neglect, so Lena and I are starting a v
    [m141] The window plants are dying of neglect, so Dana and I are starting a v
- ['m211', 'm177'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m211] The window plants are dying of neglect, so Will and I are starting a v
    [m177] The window plants are dying of neglect, so Lena and I are starting a v

Complete-story by type with relation-aware selection:

| budget | entity rel | topic rel | entity lookup | topic lookup |
| --- | --- | --- | --- | --- |
| 1000 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2000 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4000 | 1.00 | 1.00 | 1.00 | 1.00 |

## v2 (relationship 9, lookup 8; heuristic fallbacks 11)

UPDATES edge counts: entity_only 18, topic_0.50 23, topic_0.60 23, topic_0.70 23

Decision pairs:
- database ['m8', 'm90']: topic_similarity=0.23 topics=['database choice', 'moving the main store to Postgres']
    entity_only link: None
    topic@0.60 link: None
    clears threshold: {'0.50': False, '0.60': False, '0.70': False}
- hosting ['m29', 'm238']: topic_similarity=0.297 topics=['deployment platform', 'migrating app hosting to AWS']
    entity_only link: {'reason': 'entity', 'shared_entity': 'heroku', 'similarity': None}
    topic@0.60 link: {'reason': 'entity', 'shared_entity': 'heroku', 'similarity': None}
    clears threshold: {'0.50': False, '0.60': False, '0.70': False}

Eight decision-pair messages (is_decision, topic):
- m8: is_decision=True topic='database choice'
- m90: is_decision=True topic='moving the main store to Postgres'
- m29: is_decision=True topic='deployment platform'
- m238: is_decision=True topic='migrating app hosting to AWS'

Topic-created UPDATES edges at 0.60 (for false-link review): 5
- ['m67', 'm23'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m67] The window plants are dying of neglect, so Mei and I are starting a ve
    [m23] The window plants are dying of neglect, so Will and I are starting a v
- ['m117', 'm67'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m117] The window plants are dying of neglect, so Noah and I are starting a v
    [m67] The window plants are dying of neglect, so Mei and I are starting a ve
- ['m152', 'm117'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m152] The window plants are dying of neglect, so Dana and I are starting a v
    [m117] The window plants are dying of neglect, so Noah and I are starting a v
- ['m186', 'm152'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m186] The window plants are dying of neglect, so Lena and I are starting a v
    [m152] The window plants are dying of neglect, so Dana and I are starting a v
- ['m220', 'm186'] sim=1.0 topics=['watering rotation', 'watering rotation']
    [m220] The window plants are dying of neglect, so Will and I are starting a v
    [m186] The window plants are dying of neglect, so Lena and I are starting a v

Complete-story by type with relation-aware selection:

| budget | entity rel | topic rel | entity lookup | topic lookup |
| --- | --- | --- | --- | --- |
| 250 | 0.44 | 0.44 | 1.00 | 1.00 |
| 500 | 0.56 | 0.56 | 1.00 | 1.00 |
| 1000 | 0.78 | 0.78 | 1.00 | 1.00 |
| 2000 | 0.89 | 0.89 | 1.00 | 1.00 |
| 4000 | 1.00 | 1.00 | 1.00 | 1.00 |
