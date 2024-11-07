import joblib
from model import model_path
from model import vectorizer_path

nb = joblib.load(model_path)


vect = joblib.load(vectorizer_path)

message = input("请输入待被检测的短信：")

message = [message]

X_new_dtm = vect.transform(message)

y_pred = nb.predict(X_new_dtm)

match y_pred:
    case 0:
        print("这是一条正常短信，请尽快处理！")
    case 1:
        print("这是一条垃圾信息，请删除")




