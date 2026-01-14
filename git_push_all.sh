#!/bin/bash
set -e

MSG=${1:-"Update app"}

git add .
git commit -m "$MSG"
git push
