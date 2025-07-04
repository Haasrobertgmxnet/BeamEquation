#!/bin/bash

INPUT="$1"
OUTPUT="${INPUT%.tex}.md"
TEMP="tmp_cleaned.tex"

# LaTeX-Befehle ersetzen
sed -e 's/\\Meter/\\mathrm{m}/g' \
    -e 's/\\Newton/\\mathrm{N}/g' \
    "$INPUT" > "$TEMP"

# Pandoc-Konvertierung mit verschiedenen Optionen
pandoc "$TEMP" -o "$OUTPUT" \
  --from=latex \
  --to=markdown+tex_math_dollars+fenced_code_blocks \
  --wrap=preserve \
  --standalone

# Cleanup
rm "$TEMP"

# Post-processing: Verbessere die Mathe-Darstellung
sed -i -e 's/\$\$/\n$$\n/g' \
       -e 's/\$\([^$]*\)\$/`$\1$`/g' "$OUTPUT"

echo "✅ Konvertiert: $OUTPUT"
echo "💡 Für bessere Mathe-Darstellung verwende einen Viewer mit MathJax-Support"