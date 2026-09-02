# WakeWord Studio Night Shift Report

Status: RUNNING — waiting for Model A v3 sequence formal training.

## MODEL A V3

- Training status: RUNNING after one strict resume from checkpoint 500.
- Recovery note: fixed JSON serialization of NumPy `int64` validation group counts; regression test added. No training data/model/objective change.
- Best step so far: 500
- Best Validation sequence F1 so far: 0.642667
- Test access during training: none

## MODEL B

Pending Model A freeze.

## A VS B

Pending both frozen evaluations.

## SOFTWARE

Pending post-model core runtime audit.

## ESP32

- Build status: PENDING — ESP-IDF is not installed.
- Hardware status: PENDING — no device is available.

## BLOCKERS

- ESP32-S3 build and hardware test require external tooling/hardware.

## NEXT ACTION

1. Finish Model A v3 training and frozen evaluation.
2. Freeze Model A and run the bounded Model B preflight.
3. Complete A/B comparison and core runtime tests.
