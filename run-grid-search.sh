#!/bin/sh

if [ -z "$1" ]; then
	echo "No data_root given!"
	exit 1
fi

if [ -z "$2" ]; then
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

python main.py "$4" 0 iou 0.1 h false nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000001 &&
python main.py "$4" 2 iou 0.01 greedy true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000002 &&
python main.py "$4" 2 iou 0.1 greedy true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000003 &&
python main.py "$4" 2 iou 0.25 greedy true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000004 &&
python main.py "$4" 2 m 11 h true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000005 &&
python main.py "$4" 0 m 11 greedy true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000006 &&
python main.py "$4" 2 m 11 greedy false nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000007 &&
python main.py "$4" 2 m 11 greedy true nuscenes "$1" "$2" "$3"StanfordIPRL-TRI/"$4"/000008 &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000001/probabilistic_tracking_results.json \
 --version "$4" \
  --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000001  > "$3"StanfordIPRL-TRI/"$4"/000001/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000002/probabilistic_tracking_results.json \
 --version "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000002  > "$3"StanfordIPRL-TRI/"$4"/000002/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000003/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000003 > "$3"StanfordIPRL-TRI/"$4"/000003/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000004/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000004  > "$3"StanfordIPRL-TRI/"$4"/000004/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000005/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000005 > "$3"StanfordIPRL-TRI/"$4"/000005/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000006/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000006 > "$3"StanfordIPRL-TRI/"$4"/000006/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000007/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000007 > "$3"StanfordIPRL-TRI/"$4"/000007/output.txt &&

python evaluate_nuscenes.py \
 "$1" \
 "$3"StanfordIPRL-TRI/"$4"/000008/probabilistic_tracking_results.json \
 --version "$4" \
 --eval_set "$4" \
 --output_dir "$3"StanfordIPRL-TRI/"$4"/000008 > "$3"StanfordIPRL-TRI/"$4"/000008/output.txt
