#!/bin/bash

source ~/.zsh    # ggf. auskommentieren

#----------------------------------------------------
# Hier muss der Pfad angepasst werden ...
#
. /Users/mkaempf/opt/anaconda3/bin/activate && conda activate /Users/mkaempf/opt/anaconda3/envs/die-pa2;

#pip3 install -r requirements.txt

#pip3 install -U langsmith

export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=ls__d97f782cab824a37b02ee3bef05bc1bc

# The below examples use the OpenAI API, though it's not necessary in general
# export OPENAI_API_KEY=...

# Use single PDF to ask questions ...
#python3 01_RetrievalQA.py

# Create the index ... with multiple files ...
#python3 02_Docpool.py

python3 03_Tracing.py
