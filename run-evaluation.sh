#!/bin/sh

if [ -z "$1" ]; then
	echo "No data_root given!"
	exit 1
fi

if [ -z "$2" ]; then
	echo "No dataset version given!"
	exit 1
fi

if [ -z "$3" ]; then
	echo "No probabilistic_tracking_results given!"
	exit 1
fi

for results in ${3:+"$@"}
do
    echo "Evaluation of $results ..."
    python evaluate_nuscenes.py \
     "$1" \
     "$results" \
     --version "$4" \
     --output_dir "$3"StanfordIPRL-TRI/"$4"/000001  \
     > "$3"StanfordIPRL-TRI/"$4"/000001/output.txt
done