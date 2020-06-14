import numpy as np      # Library for handling arrays
import pandas as pd     # Library for data manipulation

# GUI Libraries
import tkinter as DiseasePredictor      # Interface for GUI Toolkit
from PIL import Image, ImageDraw, ImageTk
from tkinter.constants import *
from tkinter import Canvas

import csv      # Library for accessing CSV files

# Scikit Classifiers
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC                             # Support Vector Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ----------------------------------------------------------------------------------------------------------------------

# Input Training and Testing Data files
train = pd.read_csv("Training.csv")
test = pd.read_csv("Testing.csv")

diseases = test["prognosis"]    # Array containing list of diseases
diseases = np.array(diseases)

x_train = np.array(train.drop(columns=["prognosis"]))
y_train = np.array(train["prognosis"])

x_test = np.array(test.drop(columns=["prognosis"]))
y_test = np.array(test["prognosis"])

with open("Training.csv", "r") as f:
    d_reader = csv.DictReader(f)
    symptoms = d_reader.fieldnames


l2 = []
for x in range(0, len(symptoms) - 1):
    l2.append(0)
df = pd.read_csv("Training.csv")

df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
                          'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)


def Voting():
    # group / ensemble of models
    estimator = [('KNN', KNeighborsClassifier()),
                 ('SVC', SVC(gamma='auto', probability=True)),
                 ('DTC', DecisionTreeClassifier()),
                 ('MLP',
                  MLPClassifier(hidden_layer_sizes=(300,), learning_rate_init=0.001, solver='adam', random_state=42,
                                verbose=True)),
                 ('GNB', GaussianNB()),
                 ('RF', RandomForestClassifier())]

    # Voting Classifier with soft voting
    vot_soft = VotingClassifier(estimators=estimator, voting='soft')
    vot_soft.fit(x_train, y_train)
    y_pred = vot_soft.predict(x_test)

    # using accuracy_score
    score = accuracy_score(y_test, y_pred)
    print("Soft Voting Score=% d" % score)

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    for k in range(0, len(symptoms)):
        for z in psymptoms:
            if z == symptoms[k]:
                l2[k] = 1

    predict = vot_soft.predict([l2])
    predicted = predict[0]

    t1.delete("1.0", END)
    t1.insert(END, predicted)


# GUI DATA--------------------------------------------------------------------------------------------------------------

basestring = str


def hex2rgb(str_rgb):
    try:
        rgb = str_rgb[1:]

        if len(rgb) == 6:
            r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
        elif len(rgb) == 3:
            r, g, b = rgb[0] * 2, rgb[1] * 2, rgb[2] * 2
        else:
            raise ValueError()
    except:
        raise ValueError("Invalid value %r provided for rgb color." % str_rgb)

    return tuple(int(v, 16) for v in (r, g, b))


class GradientFrame(Canvas):

    def __init__(self, master, from_color, to_color, width=None, height=None, orient=HORIZONTAL, steps=None, **kwargs):
        Canvas.__init__(self, master, **kwargs)
        if steps is None:
            if orient == HORIZONTAL:
                steps = height
            else:
                steps = width

        if isinstance(from_color, basestring):
            from_color = hex2rgb(from_color)

        if isinstance(to_color, basestring):
            to_color = hex2rgb(to_color)

        r, g, b = from_color
        dr = float(to_color[0] - r) / steps
        dg = float(to_color[1] - g) / steps
        db = float(to_color[2] - b) / steps

        if orient == HORIZONTAL:
            if height is None:
                raise ValueError("height can not be None")

            self.configure(height=height)

            if width is not None:
                self.configure(width=width)

            img_height = height
            img_width = self.winfo_screenwidth()

            image = Image.new("RGB", (img_width, img_height), "#FFFFFF")
            draw = ImageDraw.Draw(image)

            for i in range(steps):
                r, g, b = r + dr, g + dg, b + db
                y0 = int(float(img_height * i) / steps)
                y1 = int(float(img_height * (i + 1)) / steps)

                draw.rectangle((0, y0, img_width, y1), fill=(int(r), int(g), int(b)))
        else:
            if width is None:
                raise ValueError("width can not be None")
            self.configure(width=width)

            if height is not None:
                self.configure(height=height)

            img_height = self.winfo_screenheight()
            img_width = width

            image = Image.new("RGB", (img_width, img_height), "#FFFFFF")
            draw = ImageDraw.Draw(image)

            for i in range(steps):
                r, g, b = r + dr, g + dg, b + db
                x0 = int(float(img_width * i) / steps)
                x1 = int(float(img_width * (i + 1)) / steps)

                draw.rectangle((x0, 0, x1, img_height), fill=(int(r), int(g), int(b)))

        self._gradient_photoimage = ImageTk.PhotoImage(image)

        self.create_image(0, 0, anchor=NW, image=self._gradient_photoimage)


if __name__ == "__main__":
    root = DiseasePredictor.Tk()
    # Symptoms selection variables
    Symptom1 = DiseasePredictor.StringVar()
    Symptom1.set(None)
    Symptom2 = DiseasePredictor.StringVar()
    Symptom2.set(None)
    Symptom3 = DiseasePredictor.StringVar()
    Symptom3.set(None)
    Symptom4 = DiseasePredictor.StringVar()
    Symptom4.set(None)
    Symptom5 = DiseasePredictor.StringVar()
    Symptom5.set(None)

    # Headings
    GradientFrame(root, from_color="#E91E63", to_color="#ffccff", height=1000, width=1000).place(height=1000)

    w2 = DiseasePredictor.Label(root, text="Disease Predictor", fg="#FCFBFB", bg="#E91E63", font=("Arial", 20, "bold"),
                                bd=5, relief=RIDGE)
    w2.grid(row=1, column=1, padx=10, sticky=W)

    w2 = DiseasePredictor.Label(root, justify=CENTER, text="A Project by Trois Chiiffres", fg="#FCFBFB", bg="#E91E63",
                                font=("Arial", 15, "bold"), bd=5, relief=RIDGE)
    w2.grid(row=3, column=1, sticky=W)

    # Labels
    S1Lb = DiseasePredictor.Label(root, text="Symptom 1", fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                                  relief=RIDGE)
    S1Lb.grid(row=7, column=0, padx=15, pady=5, sticky=W)

    S2Lb = DiseasePredictor.Label(root, text="Symptom 2", fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                                  relief=RIDGE)
    S2Lb.grid(row=8, column=0, padx=15, pady=5, sticky=W)

    S3Lb = DiseasePredictor.Label(root, text="Symptom 3", fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                                  relief=RIDGE)
    S3Lb.grid(row=9, column=0, padx=15, pady=5, sticky=W)

    S4Lb = DiseasePredictor.Label(root, text="Symptom 4", fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                                  relief=RIDGE)
    S4Lb.grid(row=10, column=0, padx=15, pady=5, sticky=W)

    S5Lb = DiseasePredictor.Label(root, text="Symptom 5", fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                                  relief=RIDGE)
    S5Lb.grid(row=11, column=0, padx=15, pady=5, sticky=W)

    # Selected values for Symptoms
    OPTIONS = sorted(symptoms)

    S1En = DiseasePredictor.OptionMenu(root, Symptom1, *OPTIONS)
    S1En.grid(row=7, column=1, padx=15, sticky=W)

    S2En = DiseasePredictor.OptionMenu(root, Symptom2, *OPTIONS)
    S2En.grid(row=8, column=1, padx=15, sticky=W)

    S3En = DiseasePredictor.OptionMenu(root, Symptom3, *OPTIONS)
    S3En.grid(row=9, column=1, padx=15, sticky=W)

    S4En = DiseasePredictor.OptionMenu(root, Symptom4, *OPTIONS)
    S4En.grid(row=10, column=1, padx=15, sticky=W)

    S5En = DiseasePredictor.OptionMenu(root, Symptom5, *OPTIONS)
    S5En.grid(row=11, column=1, padx=15, sticky=W)

    # Compute Button
    dst = DiseasePredictor.Button(root, text="Compute", command=Voting, fg="#FCFBFB", bg="#E91E63",
                                  font=("Arial", 15, "bold"), bd=5, relief=RAISED)
    dst.grid(row=15, column=0, padx=15, pady=5, sticky=W)

    # text Field
    t1 = DiseasePredictor.Text(root, height=1, width=30, fg="#E91E63", bg="#FCFBFB", font=("Arial", 15, "bold"), bd=5,
                               relief=RIDGE)
    t1.grid(row=15, column=1, padx=15, pady=5, sticky=W)

    root.mainloop()
