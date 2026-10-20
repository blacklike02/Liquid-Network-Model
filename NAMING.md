# Echo and the Pantheon family

Naming decision: 20 October 2026.

**Echo** is the chosen name of the current experimental liquid recurrent
language model. **Pantheon** is the name of its planned model family.

## Planned progression

| Tier | English name | Russian name | Intended milestone |
| --- | --- | --- | --- |
| I | **Echo** | Эхо | The initial model and its language-learning foundation |
| II | **Iris** | Ирида | Stronger dialogue and context handling |
| III | **Hermes** | Гермес | Reliable use of knowledge, memory and tools |
| IV | **Athena** | Афина | Stronger reasoning, planning and complex task completion |
| V | **Hera** | Гера | The flagship tier of the family |

These are project-defined stages inspired by mythology, not a historical
ranking of mythological figures. Future tiers are naming plans, not released
models or claims about capabilities already achieved. Hermes may have a feminine
persona; the final planned tier is Hera.

## Naming and versions

- Write the current model name as **Echo**, not an invented acronym.
- Improvements within a tier can use versions such as `Echo 1.0` and `Echo 2.0`.
  These are naming examples, not release announcements.
- A new tier name should represent a demonstrated capability improvement.
  Increasing parameters alone does not qualify a model for the next tier.
- There is no fixed parameter ceiling. Parameter counts can be reported
  separately from tier and version names.
- The existing `liquid_lm` package and `liquid_text_model.py` entry point remain
  the implementation names. Public model naming does not require changing
  imports, checkpoint tensor names or CLI commands.
- The README's existing software version describes the implementation; it does
  not automatically assign a release version to Echo.
