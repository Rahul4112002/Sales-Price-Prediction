from django.shortcuts import render
import pickle
import numpy as np

model = pickle.load(open("prediction/models/linear_regression_model.pkl","rb"))

def prediciton(request):
    result = 0
    if request.method == 'POST':
        tv = float(request.POST.get('tv'))
        radio = float(request.POST.get('radio'))
        news_paper = float(request.POST.get('news_paper'))
        
        features = np.array([[tv,radio,news_paper]])
        pred = model.predict(features)
        
        result = round(pred[0],2)
    return render(request, 'home.html',{'tv':tv,'radio':radio,'news_paper':news_paper,'output':result })
