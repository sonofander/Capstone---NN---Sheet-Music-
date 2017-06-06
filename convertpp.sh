#!/bin/bash

a=0
for f in *.pdf; do
  echo $a, ${f%.pdf}.png
  convert "${f}" -density 600 -crop 300x300+0+0 "${f%.pdf}.png"
  a=$((a+1))

done
