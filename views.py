# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



import string
import random
import os


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

import pandas
path_data=settings.MEDIA_ROOT + "//" + 'Training.csv'
data=pandas.read_csv(path_data)
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
data['prognosis']=lb.fit_transform(data['prognosis'])

x=data.iloc[:,0:-1]
print(x.columns)
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def algorithms(request):        

    # =============================================Naive Beyas=======================================

    
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        y_ = classifier.predict(x_test)
        a=accuracy_score(y_,y_test)
        b=precision_score(y_,y_test, average='weighted')
        c=recall_score(y_,y_test, average='weighted')
        d=f1_score(y_,y_test, average='weighted')
        print('Cofusion matrix for Naive Beyas')
        print('accuracy score for  is',a,b,c,d)
        cm=confusion_matrix(y_,y_test)
        print(cm)
        import matplotlib.pyplot as plt
        confusion=ConfusionMatrixDisplay(confusion_matrix=cm)
        confusion.plot()
        plt.title('Cofusion matrix for Naive Beyas')
        plt.show()
       

    # =============================================RandomForestClassifier=======================================


        from sklearn.ensemble import RandomForestClassifier
        rf=RandomForestClassifier()
        rf.fit(x_train, y_train)
        y= rf.predict(x_test)
        e=accuracy_score(y,y_test)
        f=precision_score(y,y_test, average='weighted')
        g=recall_score(y,y_test, average='weighted')
        h=f1_score(y,y_test, average='weighted')
        print('Cofusion matrix for RandomForestClassifier')
        print('accuracy score is',e,f,g,h)
        cm=confusion_matrix(y,y_test)
        print(cm)
        import matplotlib.pyplot as plt
        confusion=ConfusionMatrixDisplay(confusion_matrix=cm)
        confusion.plot()
        plt.title('Cofusion matrix for RandomForestClassifier')
        plt.show()

        

        return render(request,'users/accuracy.html',{'a':a,'b':b,'c':c,'d':d,'e':e,'f':f,'g':g,'h':h})
    
