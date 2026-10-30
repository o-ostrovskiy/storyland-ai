#!/usr/bin/env sh
# Downloads the ten public-domain novels from Project Gutenberg into ./books (not committed).
set -e
mkdir -p books
for id in 2852 1260 4300 84 768 2413 16389 41 1727 1715; do
  [ -s "books/pg$id.txt" ] || curl -sSL -A "Mozilla/5.0" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -o "books/pg$id.txt"
done
ls -l books
