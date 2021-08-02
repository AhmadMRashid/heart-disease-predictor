import pandas as pd
import tkinter as tk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class Predictor:

    def has_disease (self, row):
        self.train(self)
        return True if self.predict(self,row) == 1 else False

    @staticmethod
    def train (self):
        dataset = pd.read_csv('C:\\Users\\rashed\\Downloads\\cardio_train.csv', sep=';')
        print(dataset.head(5))
        dataset = dataset.drop('id', axis=1)
        y = dataset.cardio
        x = dataset.drop(['cardio'],axis=1)
        X_train,X_test,y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=0)
        self.classifier = RandomForestClassifier(n_estimators=12,criterion ='entropy' )
        self.classifier.fit(X_train,y_train)
        score = self.classifier.score(X_test,y_test)
        print('Train Completed -- ')
        print('Score: ' + str(score))

    @staticmethod
    def predict(self,row):
        user_df = np.array(row).reshape(1,11)
        predicted= self.classifier.predict(user_df)
        print("Predicted: " + str(predicted[0]))
        return predicted[0]

la=str()
def onClick():
    row=[[age.get(),gender.get(),height.get(),weight.get(),ap_hi.get(),ap_lo.get(),chol.get(),gluc.get(),smk.get(),alc.get(), phy.get()]]
    print(row)
    predictor = Predictor()
    o = predictor.has_disease(row)
    root2 = tk.Tk()
    root2.title("Prediction Window")
    if (o == True):
        print("Person has a heart disease")
        la="Person has a heart disease"
        tk.Label(root2, text=la, font=("times new roman", 20), fg="white", bg="maroon", height=2).grid(row=0, column=1)
    else:
        print("Person has no heart disease")
        la="Person has no heart disease"
        tk.Label(root2, text=la, font=("times new roman", 20), fg="white", bg="green", height=2).grid(row=0, column=1)

    return True


root = tk.Tk()
root.title("Heart Disease Predictor")
tk.Label(root,text="""Fill Values""",font=("times new roman", 12)).grid(row=0)

tk.Label(root,text='Age',padx=20, font=("times new roman", 12)).grid(row=1,column=0)
age = tk.IntVar()
tk.Entry(root,textvariable=age).grid(row=1,column=1)

tk.Label(root,text="""Gender""",padx=20, font=("times new roman", 12)).grid(row=2,column=0)
gender = tk.IntVar()
R1 = tk.Radiobutton(root, text="Woman", variable=gender, value=1)
R1.grid(row=2, column=1)
R2 = tk.Radiobutton(root, text="Man", variable=gender, value=2)
R2.grid(row=2, column=2)


tk.Label(root,text='Height', font=("times new roman", 12)).grid(row=3,column=0)
height = tk.IntVar()
tk.Entry(root,textvariable=height).grid(row=3,column=1)

tk.Label(root,text='Weight', font=("times new roman", 12)).grid(row=4,column=0)
weight = tk.DoubleVar()
tk.Entry(root,textvariable=weight).grid(row=4,column=1)


tk.Label(root,text='Systolic blood pressure', font=("times new roman", 12)).grid(row=5,column=0)
ap_hi = tk.IntVar()
tk.Entry(root,textvariable=ap_hi).grid(row=5,column=1)

tk.Label(root,text="""Diastolic blood pressure""",padx=20, font=("times new roman", 12)).grid(row=6,column=0)
ap_lo=tk.IntVar()
tk.Entry(root,textvariable=ap_lo).grid(row=6,column=1)

tk.Label(root,text="""Cholestrol""",padx=20, font=("times new roman", 12)).grid(row=7,column=0)
chol=tk.IntVar()
R10 = tk.Radiobutton(root, text="Normal", variable=chol, value=1)
R10.grid(row=7, column=1)
R11 = tk.Radiobutton(root, text="Above Normal", variable=chol, value=2)
R11.grid(row=7, column=2)
R12 = tk.Radiobutton(root, text="Well Above Normal", variable=chol, value=3)
R12.grid(row=7, column=3)


tk.Label(root,text="""Glucose""",padx=20, font=("times new roman", 12)).grid(row=8,column=0)
gluc=tk.IntVar()
R7 = tk.Radiobutton(root, text="Normal", variable=gluc, value=1)
R7.grid(row=8, column=1)
R8 = tk.Radiobutton(root, text="Above Normal", variable=gluc, value=2)
R8.grid(row=8, column=2)
R9 = tk.Radiobutton(root, text="Well Above Normal", variable=gluc, value=3)
R9.grid(row=8, column=3)


tk.Label(root,text='Smoking', font=("times new roman", 12)).grid(row=9,column=0)
smk = tk.IntVar()
R13 = tk.Radiobutton(root, text="Smoker", variable=smk, value=1)
R13.grid(row=9, column=1)
R14 = tk.Radiobutton(root, text="Non-Smoker", variable=smk, value=0)
R14.grid(row=9, column=2)



tk.Label(root,text="""Alcohol intake""",padx=20, font=("times new roman", 12)).grid(row=10,column=0)
alc=tk.IntVar()
R3 = tk.Radiobutton(root, text="Alcoholic", variable=alc, value=1)
R3.grid(row=10, column=1)
R4 = tk.Radiobutton(root, text="Non-Alcoholic", variable=alc, value=0)
R4.grid(row=10, column=2)


tk.Label(root,text="""Physical activity""",padx=20, font=("times new roman", 12)).grid(row=11,column=0)
phy=tk.IntVar()
R5 = tk.Radiobutton(root, text="Active", variable=phy, value=1)
R5.grid(row=11, column=1)
R6 = tk.Radiobutton(root, text="Not Active", variable=phy, value=0)
R6.grid(row=11, column=2)




tk.Label(root,text="""       """).grid(row=12)

tk.Button(root, text='Predict',font=("times new roman", 12) ,activebackground='green', 
            activeforeground='white',command=onClick).grid(row=13, column=1)
tk.Label(root,text="""       """).grid(row=14)

root.mainloop()