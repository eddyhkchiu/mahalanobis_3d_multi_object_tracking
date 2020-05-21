#!/bin/sh

if [ -z "$1" ]; then
	echo "No data_root given!"
	exit 1
fi

if [ -z " "$2"" ]; then
	echo "No detection_file given!"
	exit 1
fi

if [ -z "$3" ]; then
	echo "No output_root given!"
	exit 1
fi

if [ -z "$4" ]; then
	echo "No dataset version given!"
	exit 1
fi

timestamp=`date "+%Y%m%d-%H%M%S"`

python main.py "$4" 2 m 11 h true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/$timestamp &&
echo "Evaluation..." &&
python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/$timestamp/probabilistic_tracking_results.json \
 --version "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/timestamp  > "$3"StanfordIPRL-TRI/"$4"/$timestamp/output.txt
