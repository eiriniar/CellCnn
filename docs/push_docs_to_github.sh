#!/bin/bash

# general guide: http://daler.github.io/sphinxdoc-test/index.html

# update documentation
# make html

# update the docs page
cd ../../cellCnn_docs/html
git add .
git commit -m "rebuilt docs"
git push origin gh-pages

