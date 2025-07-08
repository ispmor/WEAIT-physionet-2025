BRANCH_NAME=$(git symbolic-ref --short HEAD | sed -r 's/\//_/g')
DT=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
LOG_FILE_NAME="logs/$BRANCH_NAME/$DT/$TIME.log"
mkdir -p logs/$BRANCH_NAME/$DT
echo $LOG_FILE_NAME

nohup bash -c 'python train_model.py -d /data/input/ -m /models/ ;python run_model.py -d /data/test/ -m /models/ -o /data/output/ ;python evaluate_model.py -d /data/test/ -o /data/output/ ' >> $LOG_FILE_NAME & 

echo "$(date)" >> $LOG_FILE_NAME 
