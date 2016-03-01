library(mlbench)
library(caret)

data("PimaIndiansDiabetes")

## creación de test y training a mano. Aunque se puede hacer con caret.

# Con caret se hace así:
control <- trainControl(method="repeatedcv", number=10, repeats=3)

# De forma general seleccionamos las filas al azar para crear el train y test asi:
# filas.entrenamiento <- trainig <- sample(1:nrow(PimaIndiansDiabetes), 0.8 * nrow(PimaIndiansDiabetes))
# PimaIndiansDiabetes.train <- PimaIndiansDiabetes[filas.entrenamiento,]
# PimaIndiansDiabetes.test  <- PimaIndiansDiabetes[-filas.entrenamiento,]

## Selección de 5 modelos a probar:

# CART: Classification and Regression Trees
set.seed(7)
fit.cart <- train(diabetes~., data=PimaIndiansDiabetes, method="rpart", trControl=control)

# LDA: Linear Discriminant Analysis
set.seed(7)
fit.lda <- train(diabetes~., data=PimaIndiansDiabetes, method="lda", trControl=control)

# SVM: Support Vector Machine with Radial Basis Function
set.seed(7)
fit.svm <- train(diabetes~., data=PimaIndiansDiabetes, method="svmRadial", trControl=control)

# KNN: k-Nearest Neighbors
set.seed(7)
fit.knn <- train(diabetes~., data=PimaIndiansDiabetes, method="knn", trControl=control)


# RF: Random Forest
set.seed(7)
fit.rf <- train(diabetes~., data=PimaIndiansDiabetes, method="rf", trControl=control)


## Recogida de resultados:
results <- resamples(list(CART=fit.cart, LDA=fit.lda, SVM=fit.svm, KNN=fit.knn, RF=fit.rf))


## Comparamos modelos (8 formas de hacerlo):

# summarize differences between modes. Miraremos Accuracy (exactitud) y Kappa.
summary(results)

# Diagrama de cajas para comparar modelos:
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

# Diagramas de desidad para Accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|")

# Digrama de puntos para accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)

# Parallel plots para comparar
parallelplot(results)

# Matriz de dispersion of predictions to compare models
splom(results)

# Comparación por pares en comparación de modelos
xyplot(results, models=c("LDA", "SVM")) # Ej: comparación modelos LDA y SVM.

# Pruebas de significación estadística
# difference in model predictions
diffs <- diff(results)
# summarize p-values for pair-wise comparisons
summary(diffs)

# Diagonal inferior muestra el p-valor par Hipótesis nula (las distribuciones son iguales). Más pequeño es mejor.
# Diagonal superior es diferencia estimada entre distribuciones.
# Lo que podemos sacar con la diag. superior es que si LDA es la mejor viendo los gráficos, podemos ver en valores absolutos cuánto es
# respecto a las otras.
