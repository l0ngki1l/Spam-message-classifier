import joblib
from train import nb
from train import vect

joblib.dump(nb, "model\\naive_bayes_model.joblib")
joblib.dump(vect, "model\\vectorizer.joblib")

model_path = "model\\naive_bayes_model.joblib"
vectorizer_path = "model\\vectorizer.joblib"