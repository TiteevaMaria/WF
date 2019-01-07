import numpy as np 
from sklearn.model_selection import train_test_split # деление данных на тренировочные и тестовые
from sklearn import metrics, linear_model # подсчет точности, линейная регрессия (тренируемая модель)

file = open("wifi_localization.txt") # исходные данные

WF = [] # качество сигнала для предсказания комнат 
RM = [] # номера комнат в соответствии с качеством сигнала WF

# первые семь значений строки - WF, сигнал, последнее - соответствующий ему номер комнаты RM
for s in file:		
    WF.append(s.split()[:7])
    RM.append(s.split()[7:])

# преобразование в нампай 
WF = np.array(WF, dtype = int)
RM = np.array(RM, dtype = int)

WF_train, WF_test, RM_train, RM_test = train_test_split(WF, RM, random_state = 0) # разбиение на тренировочные и тестовые данные

logreg = linear_model.LogisticRegression(C = 10, solver = 'lbfgs', max_iter = 10000, multi_class = 'auto') # создание модели
logreg.fit(WF_train, RM_train) # тренировка модели

pred = logreg.predict(WF_test) # отправка тестовых данных в модель
print("Accuracy:", metrics.accuracy_score(RM_test, pred)) # подсчет точности предсказания