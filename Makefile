# Stage 1.5 Pipeline — Latent Separability Audit
# Usage: make stage15
# Requires: Python 3.10+, GPU for backbone/SSL features
#
# NOTE: The primary execution environment is Google Colab via
# notebooks/stage1_5_coraa_mupe.ipynb. These Make targets provide
# an alternative local CLI using the companion repo's `stage1_5` tool.
# Scripts in scripts/ will be implemented as the pipeline matures.

PYTHON ?= python3
CONFIG ?= configs/stage1_5.yaml
SEED ?= 42

.PHONY: stage15 test clean

# Full pipeline (requires companion repo `accent-speaker-disentanglement`)
stage15:
	@echo "Run the full pipeline via: notebooks/stage1_5_coraa_mupe.ipynb (Colab)"
	@echo "Or install the companion repo and use: stage1_5 run $(CONFIG)"

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Clean generated artifacts (NOT source data)
clean:
	rm -rf reports/ data/features/ data/splits/
	@echo "Cleaned generated artifacts. Source data preserved."
