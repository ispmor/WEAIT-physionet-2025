BRANCH_NAME=$(git symbolic-ref --short HEAD | sed -r 's/\//_/g') 
LOG_FILE_NAME="logs/$BRANCH_NAME-$(date +"%Y-%m-%d_%H:%M:%S")"
echo $LOG_FILE_NAME

nohup bash -c 'python train_model.py -d /data/input/ -m /models/ ;python run_model.py -d /data/test/ -m /models/ -o /data/output/ ;python evaluate_model.py -d /data/test/ -o /data/output/ ' >> $LOG_FILE_NAME & 

echo "$(date)" >> $LOG_FILE_NAME 
