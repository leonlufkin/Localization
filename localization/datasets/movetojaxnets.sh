#!/bin/bash

# Set LC_ALL to C to avoid issues with non-ASCII characters
export LC_ALL=C

# Determine the correct sed syntax based on the OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed_cmd="sed -i ''"
else
    # Linux and others
    sed_cmd="sed -i"
fi

# Counter for modified files
modified_files=0

# Use find to locate all files, then use sed to perform the replacement
while IFS= read -r -d '' file; do
    if $sed_cmd 's/from localization\.datasets\.base import Dataset, ExemplarType/from jaxnets.datasets.base import Dataset, ExemplarType/g' "$file" 2>/dev/null; then
        if grep -q "from jaxnets.datasets.base import Dataset, ExemplarType" "$file"; then
            echo "Modified: $file"
            ((modified_files++))
        fi
    else
        echo "Error processing: $file"
    fi
done < <(find . -type f -print0)

# Clean up any backup files created by sed on macOS
find . -name "*''" -delete

echo "Replacement completed. Modified $modified_files file(s)."
echo "Cleaned up any backup files.
"
