#!/bin/bash

source ~/.zsh    # ggf. auskommentieren

#----------------------------------------------------
# Hier muss der Pfad angepasst werden ...
#
. /Users/mkaempf/opt/anaconda3/bin/activate && conda activate /Users/mkaempf/opt/anaconda3/envs/die-pa2;

pip3 install -r requirements.txt

# Use single PDF to ask questions ...
#python3 01_RetrievalQA.py

# Create the index ... with multiple files ...
#python3 02_Docpool.py
