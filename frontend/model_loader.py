import joblib

def load_clf_bow():
    """
    no input
    :return: clf and bow that were most recently trained
    """
    clf = joblib.load("../backend/models/multiNB_20-07-27T18:49:33.014222.pkl")
    bow = joblib.load("../backend/transformers/bow.pkl")
    return clf, bow


if __name__=="__main__":
    sentence = ["how are you today?"]
    vectorized = bow.transform(sentence).toarray()
    prediction = clf.predict(vectorized)
    print(vectorized)
    print(prediction)

