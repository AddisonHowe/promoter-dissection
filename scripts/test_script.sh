#!/bin/bash

seed=$1

echo "Hello bash world!"
echo "Bash got seed" $seed

python scripts/test_script.py --seed $seed

echo "Done with bash!"
