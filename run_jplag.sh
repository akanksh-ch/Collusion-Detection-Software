#!/bin/bash

# Ensure the output directory exists
mkdir -p output

# Loop strictly from 1 to 10
for i in {1..10}; do
    echo "=================================================="
    echo "Processing o${i} with JPlag..."
    echo "=================================================="
    
    # Run the exact working command structure dynamically for o1 through o10
    ~/Downloads/jdk-25/bin/java -jar ~/Downloads/jplag-6.3.0-jar-with-dependencies.jar \
        -bc "submissions/o${i}/o${i}-orig/" \
        -r "output/result-jplag-o${i}.jplag" \
        "submissions/o${i}/" \
        --mode RUN
        
    echo -e "Finished o${i}\n"
done

echo "All JPlag runs from o1 to o10 completed successfully!"
