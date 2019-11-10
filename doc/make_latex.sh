#!/bin/bash

sphinx-build -b latex . latex

cd latex

pdflatex sigpropy.tex

pdflatex sigpropy.tex
