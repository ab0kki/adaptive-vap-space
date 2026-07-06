# Baseline audit decisions

## Scope

This repository currently reports a VAP-style diagnostic baseline over VAD-derived
turn-taking events and pretrained VAP scores.

This patch does not change event extraction, event timing, labels, or VAP score
sampling. It only adds stronger metric and EDA diagnostics.

## Metrics

Weighted F1 is retained for continuity, but it is insufficient alone because
class imbalance can hide weak positive-class behavior. The evaluation now also
writes macro F1, balanced accuracy, positive precision, positive recall, positive
F1, threshold sweeps, and metrics under multiple calibration objectives.

## Event-level metrics

Frame-level metrics are retained. Event-level metrics are also written by
grouping rows by event_id and averaging scores inside the event window.

## Dataset split EDA

Dataset EDA uses canonical project split assignments from project_splits.csv or
stereo_manifest.csv. The project_split column inside interaction_manifest.csv may
be stale after incremental datastore construction and should not be treated as
canonical.

## BC-pred probability reconstruction

BC-pred event extraction is meaningful for a diagnostic baseline, but the p_bc
score is reconstructed from the 256-state VAP probability vector using a
documented approximation. BC-pred should not be described as exact paper-level
probability reconstruction until this mapping is audited against the original
implementation.

## Adaptation-relevant EDA

The EDA now emphasizes where errors occur: by event source, relative interaction
time, early versus late interaction, speaker, interaction, and interaction-level
speech statistics. These outputs are intended to support the online speaker
adaptation research question.
