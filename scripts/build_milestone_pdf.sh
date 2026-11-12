#!/usr/bin/env bash
# Build docs/milestone_report.pdf from docs/milestone_report.md (pandoc + xelatex).
# FarmShare: uses the pandoc/3.7.0.2 and texlive/20250308 module installs; elsewhere, pandoc and xelatex on PATH.
# Usage: bash scripts/build_milestone_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/../docs"
SPACK=/software/spack/opt/spack/linux-x86_64_v3
for d in "$SPACK"/pandoc-3.7.0.2-*/bin "$SPACK"/texlive-20250308-*/bin/x86_64-linux; do
    [ -d "$d" ] && PATH="$d:$PATH"
done
pandoc milestone_report.md -f commonmark_x+implicit_figures -o milestone_report.pdf --pdf-engine=xelatex --resource-path=. \
    -V geometry:margin=0.9in -V fontsize=10pt -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" \
    -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
    -V header-includes='\usepackage{etoolbox}\AtBeginEnvironment{longtable}{\footnotesize}'
echo "wrote docs/milestone_report.pdf"
