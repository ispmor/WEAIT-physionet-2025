export BRANCH_NAME=$(git symbolic-ref --short HEAD | sed -r 's/\//_/g') 
export LOG_FILE_NAME="logs/$BRANCH_NAME-TRAIN_ONLY_$(date +"%Y-%m-%d_%H:%M:%S")"
echo $LOG_FILE_NAME
mkdir -p models/$BRANCH_NAME
mkdir -p ../local_test_outputs/$BRANCH_NAME

nohup bash -c "python train_model.py -d ../training_data/all_files/ -m models/$BRANCH_NAME/ ;"
echo "$(date)" >> $LOG_FILE_NAME 
