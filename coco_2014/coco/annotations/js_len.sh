#!/bin/bash

# Function to count lines and format output
count_lines() {
    local file="$1"
    local lines=$(wc -l < "$file")
    echo "File: $(basename "$file") - Lines: $lines"
}

# Iterate over all .json files in the current directory
for json_file in *.json
do
    if [[ -f "$json_file" ]]; then
        count_lines "$json_file"
    fi
done
