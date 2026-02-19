# Stage 1.5 Pipeline — Latent Separability Audit
# Usage: make stage15
# Requires: Python 3.10+, GPU for backbone/SSL features

PYTHON ?= python3
CONFIG ?= configs/stage1_5.yaml
SEED ?= 42

.PHONY: stage15 manifest splits confounds features probes report clean test

# Full pipeline: one command to rule them all
stage15: manifest splits confounds features probes report
	@echo "Stage 1.5 pipeline complete. See reports/stage1_5_report.md"

# Step 1: Build manifest from CORAA-MUPE
manifest:
	$(PYTHON) scripts/build_manifest.py --config $(CONFIG)

# Step 2: Generate speaker-disjoint splits
splits:
	$(PYTHON) scripts/generate_splits.py --config $(CONFIG)

# Step 3: Confound analysis (accent x gender, accent x duration)
confounds:
	$(PYTHON) scripts/analyze_confounds.py --config $(CONFIG)

# Step 4: Extract all features (acoustic, ECAPA, WavLM, backbone)
features:
	$(PYTHON) scripts/extract_features.py --config $(CONFIG)

# Step 5: Run probes (accent, speaker, leakage) with CI
probes:
	$(PYTHON) scripts/run_probes.py --config $(CONFIG)

# Step 6: Generate reports
report:
	$(PYTHON) scripts/generate_report.py --config $(CONFIG)

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Clean generated artifacts (NOT source data)
clean:
	rm -rf reports/ data/features/ data/splits/
	@echo "Cleaned generated artifacts. Source data preserved."
