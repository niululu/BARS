# BARS
**Blockchain-Anchored rPPG System**

This repository contains the reference implementation and benchmarking scripts for the paper:

> **BARS: A Blockchain-Anchored Remote Photoplethysmography System for Privacy-Preserving Physiological Biometrics**  
> *(submitted to IEEE ICBC)*

BARS demonstrates a hybrid biometric architecture in which physiological templates derived from remote photoplethysmography (rPPG) are anchored on a blockchain via cryptographic commitments, while biometric processing and verification are performed entirely off-chain.

---

## Overview

### Off-chain (Python)
- POS-based rPPG extraction  
- Welch-PSD template construction and quantization  
- Enrollment–probe verification and FAR/FRR/EER evaluation  

### On-chain (Solidity / Hardhat)
- Minimal append-only registry smart contract  
- Anchors fixed-length Keccak-256 commitments with constant gas cost  

The blockchain certifies the integrity of enrolled references; authentication decisions remain off-chain.

---

## Repository Structure

```text
BARS/
├── contracts/        # Solidity registry contract
├── scripts/          # Hardhat deployment and benchmarking
├── rppg/             # Python rPPG processing and evaluation
├── figures/          # Figures used in the paper
├── hardhat.config.ts
├── package.json
├── requirements.txt
└── README.md

```
---

## Setup

### Python (off-chain processing)
``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### Node / Hardhat (on-chain benchmarking)
``` bash
npm install
npx hardhat compile
```


### Experiments
#### On-chain gas cost benchmark
``` bash
npx hardhat run scripts/benchmark.ts
```


#### Off-chain rPPG evaluation
```python
python rppg/process_ubfc.py
```


## Notes

Templates are quantized (int16) prior to hashing to ensure deterministic serialization.

Enrollment uses multi-segment template-level median fusion.

Verification aggregates distances across sliding probe windows.

No raw biometric data is stored on-chain.

## Disclaimer

This code is provided for research and experimental evaluation only.
