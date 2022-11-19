def get_prediction(record):
    
    initialize_dataset()
    
    X_test = np.array([record.age, record.sex, record.cp, record.trestbps, record.chol, record.fbs, record.restecg, \
        record.thalach, record.exang, record.oldpeak, record.slope, record.ca, record.thal])
    X_test = X_test.reshape((1,-1))
    pickled_model = pickle.load(open('prediction/model.pkl', 'rb'))
    
    val = pickled_model.predict(X_test)
    print(val)
    tar= val*100
    print(tar)
    add_data(X_test, int(val), df)
    #add data to the csv
    #initiate training with new data
    record.target = int(tar)
    return record