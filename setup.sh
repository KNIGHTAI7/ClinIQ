#!/bin/bash
# setup.sh — Streamlit Cloud pre-boot script
# Downloads the scispaCy BC5CDR medical NER model (disease + drug NER)
# This runs BEFORE the app starts, so the model is ready on first load.

pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
