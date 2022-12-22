import joblib as jb
import warnings
warnings.filterwarnings(action='ignore')

def model_call(model_file,X):
    filename = model_file
    model_dev = jb.load(filename)
    return model_dev.predict(X)

alcohol_dict={0:'Non-User',1:'Used_Over_Decade',2:'User'}
def fetch_result(result):
    return alcohol_dict[result]

