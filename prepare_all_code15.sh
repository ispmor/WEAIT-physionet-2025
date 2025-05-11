max=17
for i in `seq 1 $max`
do
	python prepare_code15_data.py -i "../data/CODE-15/hdf5/exams_part$i.hdf5" -d ../data/CODE-15/exams.csv -l ../data/CODE-15/code15_chagas_labels.csv -o "../data/CODE-15/exams_part$i"
done
