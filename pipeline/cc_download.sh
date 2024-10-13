#!/bin/bash

module load parallel

paths_file="${paths_file:-$1}"
export base_dir="${PDC_TMP}common-crawl"

dl() {
    local path="$1"
    local max_attempts=1000
    local attempt=0
    local max_backoff_seconds=60

    if [ -f "$base_dir/$path" ]; then
        # Assume file is downloaded and valid if it exists
        return 0
    fi

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        status=$(curl "https://data.commoncrawl.org/$path" -w "%{http_code}\n" --retry 10 --create-dirs --silent --output "$base_dir/$path.tmp")
        
        # Check for 403 and also verify file is valid (in case of broken downloads) 
        if [[ "$status" -ne 403 ]]; then
            # If it's a gzipped file, we also verify its validity
            if ! [[ "$path" == *.gz ]] || gzip -t "$base_dir/$path.tmp"; then
                mv "$base_dir/$path.tmp" "$base_dir/$path"
                return 0
            fi
        fi
        echo "$(date '+%Y-%m-%d %H:%M:%S') WARNING: Attempt $attempt/$max_attempts - $path"

        # Remove file if invalid
        rm -f "${PDC_TMP}common-crawl/$path.tmp"
        sleep $(( 2**$attempt > $max_backoff_seconds ? $max_backoff_seconds : 2**$attempt))
    done

    echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: Failed after $max_attempts attempts with status $status - $path"
    return 1
}

export -f dl

# Download paths file
if [ ! -f "$base_dir/$paths_file" ]; then
    dl "$paths_file"
fi

# Use GNU parallel to run requests in parallel
gzip -cd $base_dir/$paths_file | parallel --lb -j 300 --eta --progress dl