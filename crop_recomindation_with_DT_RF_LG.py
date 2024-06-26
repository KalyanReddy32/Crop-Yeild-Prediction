from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import root
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def open_second_page():
    second_window = tk.Toplevel(main)
    second_window.title("Second Page")
    second_window.geometry("500x500")
    second_window.config(bg='#FFDAB9')

    label1 = tk.Label(second_window, font=("times", 15), text="Enter The Values For Prediction")
    label1.pack(pady=10)

    label_temperature = tk.Label(second_window,font=("times",15), text="Temperature:")
    label_temperature.pack(pady=10)

    input_temperature = tk.Entry(second_window,font=("times",15),)
    input_temperature.pack(pady=10)

    label_humidity = tk.Label(second_window,font=("times",15), text="Humidity:")
    label_humidity.pack(pady=10)

    input_humidity = tk.Entry(second_window,font=("times",15))
    input_humidity.pack(pady=10)

    label_ph = tk.Label(second_window,font=("times",15), text="pH Value:")
    label_ph.pack(pady=10)

    input_ph = tk.Entry(second_window,font=("times",15))
    input_ph.pack(pady=10)

    label_rainfall = tk.Label(second_window,font=("times",15), text="Rainfall:")
    label_rainfall.pack(pady=10)

    input_rainfall = tk.Entry(second_window,font=("times",15))
    input_rainfall.pack(pady=10)

    def submit_second_page():
        global input_data,temperature, humidity, ph_value, rainfall

        temperature = input_temperature.get()
        humidity = input_humidity.get()
        ph_value = input_ph.get()
        rainfall = input_rainfall.get()

        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'Humidity': ["    "+humidity],
            'pH': [ph_value],
            'Rainfall': [rainfall]
        })
        text.delete('1.0', END)

        text.insert(END,input_data)
        print(input_data)

        second_window.destroy()

        # You can perform other actions with the input values here

    submit_button = tk.Button(second_window, text="Submit", command=submit_second_page, bg="turquoise",width=10)
    submit_button.pack(pady=10)

global filename
global df, X_train, X_test, y_train, y_test

# Upload Function
def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")


# Prediction function
def prediction(X_test, cls):
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" (X_test[i], y_pred[i]))
    return y_pred



# Split the dataset
def splitdataset(): 
    global df, X_train, X_test, y_train, y_test
    X = df[['temperature', 'humidity', 'ph', 'rainfall']]

    y = df['label']
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train.shape)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test.shape)) + "\n\n")
    text.insert(END, str(X))
    text.insert(END, str(y))
    return X, y, X_train, X_test, y_train, y_test


#Decision Tree Function
def dt():
    global X_train, y_train, X_test, y_test
    global Decision,dt_acc
    text.delete('1.0', END)
    Decision = DecisionTreeClassifier(criterion='entropy', random_state=0)
    Decision.fit(X_train, y_train)
    text.insert(END, "Prediction Results\n\n")
    y_pred11 = Decision.predict(X_test)
    dt_acc = accuracy_score(y_test, y_pred11)
    print('Accuracy for the decision Tree is ', dt_acc*100,'%')
    text.insert(END, "DT Accuracy : " + str(dt_acc*100) + "\n\n")

#Random Forest Function
def rand():
    global Random,random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    Random = RandomForestClassifier(n_estimators=10, criterion="entropy")
    Random.fit(X_train, y_train)
    text.insert(END,"Prediction Results\n\n")
    y_pred2=Random.predict(X_test)
    random_acc = accuracy_score(y_test, y_pred2)
    print('Accuracy for the Random Forest is ', random_acc * 100, '%')
    text.insert(END,"Random Accuracy : "+str(random_acc*100)+"\n\n")

#Lodistic Regression Function
def logisticRegression():
    global lr_accuracy
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    text.delete('1.0', END)
    text.insert(END,"Prediction Results\n\n")
    y_pred = model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy logistic regression: {lr_accuracy * 100}")
    text.insert(END,"Accuracy logistic regression : "+str(lr_accuracy*100)+"\n\n")

import matplotlib.pyplot as plt

def accuracy_graph():
    algorithms = ['Random Forest', 'Logistic Regression', 'Decision Tree']
    accuracy_scores = [random_acc * 100, lr_accuracy * 100, dt_acc * 100]

    plt.bar(algorithms, accuracy_scores, color=['blue', 'green', 'red'])
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 100)

    # Add text labels on top of the bars
    for i in range(len(algorithms)):
        plt.text(i, accuracy_scores[i], f'{accuracy_scores[i]:.2f}%', ha='center', va='bottom')

    plt.show()


def predict():
    l1=LabelEncoder()

    records = input_data.values[:, 0:4]
    print("===>", records)

    value = Random.predict(records)
    print("result of Random Tree:" + str(value))
    text.insert(END,"\n\n")
    text.insert(END, "Result Of Random Forest: " + str(value) + "\n\n")

main = tk.Tk()
main.title("Big Data Analysis Technology Application in Agricultural Intelligence Decision System") 
main.geometry("1600x1500")

font = ('times', 16, 'bold')
title = Label(main, text='Big Data Analysis Technology Application in Agricultural Intelligence Decision System',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)


font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=180)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
# Upload Crop Recommendation button with color sky blue
uploadButton = Button(main, text="Upload Crop Recommendation", command=upload, bg="sky blue")
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)



pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=330, y=550)

# Split Dataset button with color light green
splitButton = Button(main, text="Split Dataset", command=splitdataset, bg="light green")
splitButton.place(x=50, y=650)
splitButton.config(font=font1)

# Decision Tree button with color turquoise
dtButton = Button(main, text="Decision Tree", command=dt, bg="turquoise")
dtButton.place(x=200, y=650)
dtButton.config(font=font1)

# RF Algorithm button with color coral
ranButton = Button(main, text="Run RF Algorithm", command=rand, bg="coral")
ranButton.place(x=350,y=650)
ranButton.config(font=font1)

# Logistic Regression button with color gold
LRButton = Button(main, text="Logistic Regression", command=logisticRegression, bg="gold")
LRButton.place(x=530,y=650)
LRButton.config(font=font1)

# Show Accuracy Graph button with color violet
accuracy_graph_button = Button(main, text="Show Accuracy Graph", command=accuracy_graph, bg="violet")
accuracy_graph_button.place(x=720, y=650)
accuracy_graph_button.config(font=font1)

# Enter The Values For Prediction button with color green
open_second_button = tk.Button(main,font=(13), text="Enter The Values For Prediction", command=open_second_page, bg="green")
open_second_button.place(x=950, y=650)
open_second_button.config(font=font1)

# Prediction button with color orange
open_second_button = tk.Button(main,font=(13), text="Prediction", command=predict, bg="orange")
open_second_button.place(x=1250, y=650)
open_second_button.config(font=font1)

main.config(bg='#F08080')
main.mainloop()