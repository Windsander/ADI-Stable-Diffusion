#!/usr/bin/env sh

# for Mac temp files cleaning
find . -name "._*" -type f -print
find . -name "._*" -type f -delete