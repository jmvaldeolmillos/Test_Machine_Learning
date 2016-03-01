
## http://machinelearningmastery.com/machine-learning-evaluation-metrics-in-r/

### EN CARET lo básico:
    # Para problemas de clasificacion se usará: Accuracy
    # Para modelos de Regresión se usarán: RMSE

# Pero también podemos usar en Caret otros:
    # Accuracy and KAPPA
    # RMSE and R^2
    # ROC (AUC, sensitivity and Specificity)
    # LogLoss



## 1 ACCURACY AND KAPPA: para conjuntos de datos binarios y clasificación con varias categorias.
    # Accuracy: % de casos correctamente clasificados. Util en una clasificación binaria.
    # Kappa: Precisión de clasificación. Útil en lso problemas con desequilibrio en las clases (Ej: 70-30)

library(caret)
library(mlbench)

# load the dataset
data(PimaIndiansDiabetes)

# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)
fit <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", metric="Accuracy", trControl=control)

# display results
print(fit)
# muestra accuracy de 77% no muy alto (el corte inicial estaba en 65% neg y 35% Posit.) y kappa de 46%, mejor.



## 2 RMSE y R^2
    # RMSE nos da una idea de cuanto de bien o mal lo está haciendo un algoritmo.
    # R^2: Bondad de ajuste entre 0 no compatible y 1 ajuste perfecto.

library(caret)

# load data
data(longley)

# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)
fit <- train(Employed~., data=longley, method="lm", metric="RMSE", trControl=control)

# display results
print(fit)
# RMSE un 0,38
# R^2 muy alto con 0.988



## 3 Area bajo la curva de ROC
    # Muestra la capacidad para discriminar. un area de 1 representa un modelo que predice 100%. Uno de 0.5 como si fuera azar.
    # Sensibilidad: Verdadera tasa positiva: Número de casos positivos que se predicen correctamente.
    # Especificidad: Verdadera tasa negativa: Numero de casos negativos que se predicen correctamente.

# load libraries
library(caret)
library(mlbench)

# load the dataset
data(PimaIndiansDiabetes)

# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)
set.seed(7)
fit <- train(diabetes~., data=PimaIndiansDiabetes, method="glm", metric="ROC", trControl=control)

# display results
print(fit)
# Se puede ver un valor de 0.833 que es bueno, pero no excelente.



## 4 Pérdida logarítmica
    # LogLoss: Para clasificación binaria, aunque también con varias categorías.
    # Evalúa probabilidades estimadas por los algoritmos.

# load libraries
library(caret)

# load the dataset
data(iris)

# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)
set.seed(7)
fit <- train(Species~., data=iris, method="rpart", metric="logLoss", trControl=control)

# display results
print(fit)
# El LogLoss es mínimo. El modelo óptimo CART tuvo un cp de 0