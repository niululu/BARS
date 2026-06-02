# BARS
**Blockchain-Anchored rPPG System**

This repository contains the reference implementation and benchmarking scripts for the paper:

> **BARS: A Blockchain-Anchored System for rPPG-Based Biometric Verification**  
> *(Published in the Proceedings of IEEE ICBC)*

BARS demonstrates a hybrid biometric architecture in which physiological templates derived from remote photoplethysmography (rPPG) are anchored on a blockchain via cryptographic commitments, while biometric processing and verification are performed entirely off-chain.

---

## Overview

### Off-chain (Python)
- POS-based rPPG extraction  
- Welch-PSD template construction and quantization  
- Enrollment–probe verification and FAR/FRR/EER evaluation  

### On-chain (Solidity / Hardhat)
- Minimal append/revoke-only registry smart contract  
- Anchors fixed-length Keccak-256 commitments with constant gas cost  

The blockchain certifies the integrity of enrolled references; authentication decisions remain off-chain.

---

## Repository Structure

```text
BARS/
├── contracts/                                # Solidity smart contracts (append/revoke-only registry)
│   └── BARS.sol
│
├── scripts/                                  # Hardhat deployment and gas benchmarking scripts
│   ├── deploy.ts
│   └── benchmark.ts
│
├── rppg/                                     # Python rPPG processing and biometric evaluation
│   ├── gen_UBFC_templates.py                 # Convert original UBFC videos into npz files
│   └── ubfc_npz_to_hash.py                   # Generate rPPG templates, commitments, and verification results
│
├── utils/                                    # Shared utilities
│
├── environment.yml                           # Conda environment specification
├── requirements.txt                          # Python dependencies
│
├── hardhat.config.ts                         # Hardhat configuration
├── tsconfig.json                             # TypeScript configuration for Hardhat scripts
│
├── package.json                              # Node.js dependencies
├── package-lock.json                         # Dependency lock file
│
├── gas_results.json                          # Gas benchmarking results
│
└── README.md
```
---

## 1. Environment Setup (Python / Conda)

We recommend using **Conda** to ensure a reproducible Python environment.

Option A: Restore full Conda environment (recommended)
```bash
conda env create -f environment.yml
conda activate bars
```

Option B: Build Conda environment manually
```bash
conda create -n bars python=3.9 -y
conda activate bars
pip install -r requirements.txt
```

## 2. UBFC Data Preparation and rPPG Commitment Generation

Download the UBFC-rPPG dataset to a local directory. The dataset is not included in this repository.

Create the output directory for generated npz templates:
```bash
mkdir -p rppg/raw_templates
```
(This directory is generated locally and is not included in the repository.)

Convert the original UBFC videos into npz files with key `video`:
```bash
python rppg/gen_UBFC_templates.py -i <UBFC_DATASET_DIR> -o rppg/raw_templates
```

Generate rPPG templates, Keccak-256 commitments, and verification statistics:
```bash
python rppg/ubfc_npz_to_hash.py \
  -i rppg/raw_templates \
  -o rppg/ubfc_npz_commitments.csv \
  --within_csv rppg/ubfc_within_session_distances.csv
```
The generated CSV files are local experiment outputs and are not committed to the repository.


## 3. Node.js & Hardhat Setup (Blockchain Module)

The blockchain component is implemented using Hardhat.
Please ensure that Node.js ≥ 22 (LTS) is installed.

We recommend using nvm:
```bash
nvm install 22
nvm use 22
```

Verify the installation (recommended npm ≥ 10):
```bash
node -v
npm -v
```

Install Hardhat dependencies:
```
npm install
```

## 4. Compile Smart Contracts

Compile the Solidity smart contract:
```
npx hardhat compile
```

Expected output:
```
Compiled 1 Solidity file with solc 0.8.20
```

## 5. Launch Local Blockchain Network

Start a local Hardhat node (keep this terminal running):
```
npx hardhat node
```

This launches a local Ethereum network at:
```
http://127.0.0.1:8545
```

## 6. Deploy Smart Contract

In a new terminal, deploy the contract to the local network:
```
npx hardhat run scripts/deploy.ts --network localhost
```

The script will output the deployed contract address.

## 7. Gas Benchmarking / Registry Evaluation

Run the gas benchmarking or registry interaction scripts:
```
npx hardhat run scripts/benchmark.ts --network localhost
```


This step records the gas cost for on-chain commitment anchoring and completes the blockchain evaluation pipeline.

## 8. Notes on Reproducibility

- All blockchain experiments are conducted on a local Hardhat network.

- Gas measurements are deterministic under the local environment.

- Python-based rPPG processing and blockchain anchoring are decoupled by design.

- No GPU or external blockchain access is required.

- Templates are quantized (int16) prior to hashing to ensure deterministic serialization.

- Enrollment uses multi-segment template-level median fusion.

- Verification aggregates distances across sliding probe windows.

- No raw biometric data is stored on-chain.

## Disclaimer

This code is provided for research and experimental evaluation only.
