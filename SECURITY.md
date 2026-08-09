# Security

## Supported versions

billhook is pre-1.0. Only the latest release is supported.

| Version | Supported |
|---|---|
| 0.1.x   | yes       |
| < 0.1.0 | no        |

## Reporting a vulnerability

Use GitHub private vulnerability reporting on
[hedgerow-dev/billhook](https://github.com/hedgerow-dev/billhook):
Security tab → Report a vulnerability. There is no security email address for
this project.

We aim to acknowledge reports within 7 days.

## Scope notes

billhook downloads third-party corpora pinned by SHA-256 and feeds adversarial
text through detectors. The corpora are attack strings by construction; they
are input to classifiers and are never executed.

The trust boundaries are:

1. **The pins are the integrity control.** The SHA-256 pins in
   `src/billhook/fetch.py` are what stand between a tampered upstream and the
   bytes every published number is read off. A forged pin is the supply-chain
   attack on this project, which is why changes to `DATASETS` get maintainer
   review.
2. **Hosted adapters send corpus text to third-party APIs.** That is why they
   are opt-in specs, and why the comparison table prints
   `requires=network+key` beside them. Recall that ships prompts off the
   machine is not comparable to recall that runs air-gapped, and the table
   says so.
3. **The harness runs no model and executes nothing from the corpora.**
   `pip install billhook` pulls no detector, no model, no framework.

Deliberately out of scope:

- Vulnerabilities in the detectors billhook scores. Report those upstream to
  the detector's own project.
- The adversarial content of the corpora. Attack strings in an attack corpus
  are the point of it.
