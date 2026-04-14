# Imports are done at function call time to avoid hard torch dependency
# when only using utility functions.
#
# Usage:
#   from pipeline.tissue.model import load_tissue_model
#   from pipeline.tissue.infer import run_tissue_classification
