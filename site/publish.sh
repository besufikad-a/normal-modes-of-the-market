#!/usr/bin/env bash
# Rebuild the site data and publish docs/ to the gh-pages branch (GitHub Pages).
# Usage (repo root, after committing your changes to main):  bash site/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python site/build_data.py
cp paper/normal-modes-of-the-market.pdf animations/normal_modes_of_the_market.mp4 docs/media/
if ! git diff --quiet -- docs; then
  git add docs && git commit -m "Update site data"
  git push origin main
fi
git branch -D gh-pages 2>/dev/null || true
git subtree split --prefix docs -b gh-pages
git push --force origin gh-pages
echo "Published: https://besufikad-a.github.io/normal-modes-of-the-market/"
