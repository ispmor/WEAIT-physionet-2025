import os, shutil
import logging 



logger = logging.getLogger(__name__)
def clean_datasets_directory():
    folder = 'h5_datasets'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.warn('Failed to delete %s. Reason: %s' % (file_path, e))
