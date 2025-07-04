#!/bin/bash

INPUT="$1"
OUTPUT="${INPUT%.tex}.md"
TEMP="tmp_cleaned.tex"

# LaTeX-Befehle ersetzen und GitHub-kompatibel machen
sed -e 's/\\Meter/\\mathrm{m}/g' \
    -e 's/\\Newton/\\mathrm{N}/g' \
    -e 's/\\left/\\left/g' \
    -e 's/\\right/\\right/g' \
    "$INPUT" > "$TEMP"

# Pandoc-Konvertierung für GitHub-kompatibles Markdown
pandoc "$TEMP" -o "$OUTPUT" \
  --from=latex \
  --to=gfm+tex_math_dollars \
  --wrap=preserve

# Post-processing: GitHub-spezifische Anpassungen
sed -i -e 's/\$\$\([^$]*\)\$\$/\n```math\n\1\n```\n/g' \
       -e 's/\$\([^$]*\)\$/\$\1\$/g' \
       -e 's/```math\n\n/```math\n/g' \
       -e 's/\n\n```/\n```/g' "$OUTPUT"

# Cleanup
rm "$TEMP"

echo "✅ Konvertiert: $OUTPUT"
echo "🐙 Optimiert für GitHub-Markdown mit Math-Rendering"
echo "💡 Display-Formeln verwenden ```math Blöcke, Inline-Formeln $...$"