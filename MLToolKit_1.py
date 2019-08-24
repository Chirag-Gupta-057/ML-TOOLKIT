from tkinter import filedialog
from tkinter import *
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from pandastable import Table,TableModel
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tkinter.font import Font
#################upload file    
def get_Csv():
    chirag.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    file=chirag.filename
#################class for new window
class TestApp(Frame):
    """Basic test frame for the table"""
    def __init__(self, parent=None):
        window = Toplevel(chirag)
        f = Frame(window)
        f.pack(fill=BOTH,expand=1)
        df = TableModel.getSampleData()
        table = pt = Table(f, dataframe=df,showtoolbar=True, showstatusbar=True)
        pt.importCSV(chirag.filename)
        pt.show()
        return
###############object creation and display of new window
def show():

    app = TestApp()




####################SUPPORT VECTOR MACHINE
def svmClasifier():
    global y,random_state,test_size,reg_algo,mae,r2Score,coeff,acc
    y1=y.get()
    random_state1=random_state.get()
    test_size1=test_size.get()
    dataset = pd.read_csv(chirag.filename)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,y1].values
    
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_size1)
    algo=reg_algo.get()

    svm_model=SVC(C = 1.0, gamma = 0.1)   
    svm_model.fit(X_train,Y_train)
    svm_prediction=svm_model.predict(X_test)

    mae.set(metrics.mean_absolute_error(Y_test,svm_prediction))
    r2Score.set(r2_score(Y_test,svm_prediction))
    coeff.set(svm_model.intercept_)
    acc.set(svm_model.score(X_test,Y_test))

    plt.scatter(Y_test,svm_prediction ,color="green",marker="*",s=200)
    plt.plot(Y_test,svm_prediction ,color="orange")

    plt.show()

####################LOGISTIC REGRESSION
def logistic():
    global y,random_state,test_size,reg_algo,mae,r2Score,coeff,acc
    y1=y.get()
    random_state1=random_state.get()
    test_size1=test_size.get()
    dataset = pd.read_csv(chirag.filename)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,y1].values
    
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_size1)
    algo=reg_algo.get()
    
    
    from sklearn.linear_model import LogisticRegression
    reg_model = LogisticRegression()
    reg_model.fit(X_train,Y_train)
    Y_pred = reg_model.predict(X_test)
    
    mae.set(metrics.mean_absolute_error(Y_test,Y_pred))
    r2Score.set(r2_score(Y_test,Y_pred))
    coeff.set(reg_model.intercept_)
    acc.set(reg_model.score(X_test,Y_test))

##################LINEAR REGRESSION
def regressor():
    global y,random_state,test_size,reg_algo,mae,r2Score,coeff,acc
    y1=y.get()
    random_state1=random_state.get()
    test_size1=test_size.get()
    dataset = pd.read_csv(chirag.filename)
    X = dataset.iloc[:,:-1].values
    Y = dataset.iloc[:,y1].values
    
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=test_size1)
    algo=reg_algo.get()
    if algo=='LinearRegression':
        from sklearn.linear_model import LinearRegression
        reg_model = LinearRegression()

    
    reg_model.fit(X_train,Y_train)
    Y_pred = reg_model.predict(X_test)
    
    mae.set(metrics.mean_absolute_error(Y_test,Y_pred))
    r2Score.set(r2_score(Y_test,Y_pred))
    coeff.set(reg_model.intercept_)
    acc.set(reg_model.score(X_test,Y_test))    
#####################main Loop
chirag = Tk()
chirag.geometry("600x700+500+5")
chirag.title("ML ToolKit")
myfont = Font(family="Ubuntu Mono", size=16)

title_frame=Frame(chirag)
caption_frame=Frame(chirag)
options_frame=Frame(chirag)
preprocess_frame=Frame(chirag,pady=50)
algo_frame = Frame(chirag,pady=50)

title_frame.pack(fill=X)
#caption_frame.pack(fill=X)
options_frame.pack()

title_label=Label(title_frame,text="ML TOOLKIT",relief="sunken",font=myfont, bg='SlateGray3')
title_label.pack()


Button(options_frame,text="Upload Csv",relief="raised",command=get_Csv,font=myfont,bg="yellow").pack()
Button(options_frame,text="Show",relief="raised",command=show,font=myfont,bg="yellow").pack()

###########
preprocess_frame.pack()

y=IntVar()
random_state=IntVar()
test_size=DoubleVar()

mae = DoubleVar()
r2Score = DoubleVar()
coeff = DoubleVar()
acc = DoubleVar()

Label(preprocess_frame,text="PreProcess Data",relief=SUNKEN,font=myfont, bg='SlateGray4').grid(row=0,column=2)

Label(preprocess_frame,text="Target index[]:",relief=RIDGE,font=myfont, bg='SlateGray4').grid(row=2,column=0)
Label(preprocess_frame,text="Random State:",relief=RIDGE,font=myfont, bg='SlateGray4').grid(row=3,column=0)
Label(preprocess_frame,text=" Test Size(0-1):",relief=RIDGE,font=myfont, bg='SlateGray4').grid(row=4,column=0)


   
Entry(preprocess_frame,textvariable=y,width=10).grid(row=2,column=3)
Entry(preprocess_frame,textvariable=random_state,width=10).grid(row=3,column=3)
Entry(preprocess_frame,textvariable=test_size,width=10).grid(row=4,column=3)

algo_frame.pack()
Label(algo_frame,text="Select Algo",relief=SUNKEN,width=10,font=myfont, bg='DarkSeaGreen4').pack()

reg_frame = Frame(algo_frame)
reg_frame.pack(side=LEFT)

reg_algo=StringVar()
Label(reg_frame,text="Regression Algo's",relief=SUNKEN,font=myfont, bg='DarkSeaGreen4').grid(row=0,column=0)
Radiobutton(reg_frame,indicatoron = 0,text="LinearRegression",variable=reg_algo,value="LinearRegression",command=regressor,font=myfont,activebackground="yellow").grid(row=1,column=0)
Radiobutton(reg_frame,indicatoron = 0,text="LogisticRegression",variable=reg_algo,value="LogisticRegression",command=logistic,font=myfont,activebackground="yellow").grid(row=2,column=0)

clas_frame = Frame(algo_frame)
clas_frame.pack(side=LEFT)

clas_algo=StringVar()
Label(clas_frame,text="Classification Algo's",relief=SUNKEN,font=myfont, bg='DarkSeaGreen4').grid(row=0,column=0,pady=(10,0))
Radiobutton(clas_frame,indicatoron = 0,text="SVM",variable=clas_algo,value="SVM",command=svmClasifier,font=myfont,activebackground="yellow").grid(row=1,column=0,pady=(0,50))




result_frame = Frame(chirag)
result_frame.pack()

Label(result_frame,text="Absolute Error:",font=myfont, bg='DarkSeaGreen3').grid(row=0,column=0)
Label(result_frame,text="R2 Score:",font=myfont, bg='DarkSeaGreen3').grid(row=1,column=0)
Label(result_frame,text="Coefficient:",font=myfont, bg='DarkSeaGreen3').grid(row=2,column=0)
Label(result_frame,text="Accuracy:",font=myfont, bg='DarkSeaGreen3').grid(row=3,column=0)

Entry(result_frame,textvariable=mae,width=20).grid(row=0,column=1)
Entry(result_frame,textvariable=r2Score,width=20).grid(row=1,column=1)
Entry(result_frame,textvariable=coeff,width=20).grid(row=2,column=1)
Entry(result_frame,textvariable=acc,width=20).grid(row=3,column=1)

chirag.mainloop()

