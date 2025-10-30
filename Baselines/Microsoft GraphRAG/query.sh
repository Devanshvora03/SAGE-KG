#!/bin/bash
# GraphRAG Batch Query Processor
# Usage: ./graphrag_batch.sh questions.txt output_directory method


if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <questions_file> <output_directory> <method> <root dir> <response type>"
    echo "Method: global or local"
    exit 1
fi

QUESTIONS_FILE="$1"
OUTPUT_DIR="$2"
METHOD="$3"
ROOT_DIR="$4"
RESPONSE_TYPE="$5"

# Check if questions file exists
if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "Error: Questions file '$QUESTIONS_FILE' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Initialize counters
TOTAL_QUESTIONS=$(wc -l < "$QUESTIONS_FILE")
CURRENT=264
FAILED=0

echo "Starting batch processing of $TOTAL_QUESTIONS questions..."
echo "Method: $METHOD"
echo "Output directory: $OUTPUT_DIR"
echo "=========================="

# Read file line by line and process each question
while IFS= read -r question || [ -n "$question" ]; do
    if [ -z "$question" ]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))


    CLEAN_QUESTION=$(echo "$question" | sed 's/[^a-zA-Z0-9 ]//g' | tr ' ' '_' | cut -c1-50)
    OUTPUT_FILE="$OUTPUT_DIR/answer_${CURRENT}_${CLEAN_QUESTION}.txt"

    echo "[$CURRENT/$TOTAL_QUESTIONS] Processing: $question"

    if graphrag query \
        --root "$ROOT_DIR" \
        --method "$METHOD" \
        --query "$question" \
        --response-type "$RESPONSE_TYPE" > "$OUTPUT_FILE" 2>&1; then
        echo "✓ Answer saved to: $OUTPUT_FILE"
    else
        echo "✗ Failed to process question $CURRENT"
        echo "Error processing: $question" > "${OUTPUT_FILE}.error"
        FAILED=$((FAILED + 1))
    fi

    echo "---"

    sleep 1

done < "$QUESTIONS_FILE"

echo "=========================="
echo "Batch processing completed!"
echo "Total questions: $TOTAL_QUESTIONS"
echo "Successful: $((TOTAL_QUESTIONS - FAILED))"
echo "Failed: $FAILED"
