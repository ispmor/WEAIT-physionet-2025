LOG_FILE_NAME="logs/TRAIN_RUN_SCORE_LOG-$(date)"
echo $LOG_FILE_NAME

nohup python train_model.py -d /data/input/ -m /models/ >> $LOG_FILE_NAME & 

nohup python run_model.py -d /data/test/ -m /models/ -o /data/output/ >> $LOG_FILE_NAME &

nohup python evaluate_model.py -d /data/test/ -o /data/output/ >> $LOG_FILE_NAME &

echo "$(date)" >> $LOG_FILE_NAME 
