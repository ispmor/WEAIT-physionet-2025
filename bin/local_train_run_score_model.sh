BRANCH_NAME=$(git symbolic-ref --short HEAD | sed -r 's/\//_/g') 
LOG_FILE_NAME="logs/$BRANCH_NAME-$(date +"%Y-%m-%d_%H:%M:%S")"
echo $LOG_FILE_NAME
mkdir -p models/$BRANCH_NAME
mkdir -p ../local_test_outputs/$BRANCH_NAME

nohup bash -c 'python train_model.py -d ../training_data/all_files/ -m models/$BRANCH_NAME/ ;python run_model.py -d ../test_files/ -m models/$BRANCH_NAME/ -o ../local_test_outputs/$BRANCH_NAME/ ;python evaluate_model.py -d ../test_files/ -o ../local_test_outputs/$BRANCH_NAME/ ' >> $LOG_FILE_NAME & 

echo "$(date)" >> $LOG_FILE_NAME 
