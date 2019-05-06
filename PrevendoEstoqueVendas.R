# Projeto Final 2 - Prevendo demanda Estoque base de Vendas


setwd("~/PrevendoDemandaEstoqueBaseVendas")

getwd()


###################### CARREGANDO OS DATASETS ######################

library(data.table)

#train_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/train.csv",
#                       sep = ",",
#                       select = c("Semana","Agencia_ID","Canal_ID","Ruta_SAK","Cliente_ID","Producto_ID","Venta_uni_hoy","Dev_uni_proxima","Demanda_uni_equil"),
#                       stringsAsFactors = FALSE)

train_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/train_dataset_reduced.csv",
                       sep = ",",
                       stringsAsFactors = FALSE)


test_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/test.csv",
                      sep = ",",
                      stringsAsFactors = FALSE)

state_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/town_state.csv",
                       sep = ",",
                       stringsAsFactors = FALSE)

client_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/cliente_tabla.csv",
                        sep = ",",
                        stringsAsFactors = FALSE)

product_dataset <- fread(file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/producto_tabla.csv",
                         sep = ",",
                         stringsAsFactors = FALSE)


# Verificando se existem dados missing
any(is.na(train_dataset))

head(test_dataset)
head(train_dataset)
head(state_dataset)
head(client_dataset)
head(product_dataset)

#View(train_dataset)
#View(test_dataset)

###################### REALIZANDO ANÁLISE EXPLORATÓRIA DOS DADOS ###################### 
str(train_dataset)

library(ggplot2)
library(dplyr)

# Gráfico Barras TOP 10 produtos mais vendidos
prod_vendido <- inner_join(train_dataset, product_dataset, by = c("Producto_ID" = "Producto_ID")) %>%
  group_by(NombreProducto) %>%
  summarise(QtdByProduct = sum(Venta_uni_hoy)) 

prod_vendido %>%
  top_n(n = 10) %>%
  arrange(desc(QtdByProduct)) %>%
  ggplot(aes(x = reorder(NombreProducto,-QtdByProduct), y = QtdByProduct)) + 
  geom_bar(stat = "identity") +
  ggtitle("TOP 10 Produto mais vendido") +
  xlab("Nome do Produto") +
  ylab("Qtd Vendida") +
  ylim(c(0,115000))

# Gráfico barras de Produtos Vendidos por Semana
train_dataset %>%
  group_by(Semana) %>%
  summarise(QtdByWeek = sum(Venta_uni_hoy)) %>%
  ggplot(aes(x = Semana, y = QtdByWeek)) + 
  geom_bar(stat = "identity") +
  xlim(c(2, 10)) +
  ggtitle("Quantidade de Produtos Vendidos por Semana")  

# Gráfico barras de Regisros agrupados por semana
train_dataset %>%
  group_by(Semana) %>%
  summarise(QtdRows = n()) %>%
  ggplot(aes(x = Semana, y = QtdRows)) +
  geom_bar(stat = "identity") +
  xlim(c(2, 10)) +
  ggtitle("Quantidade de Registros/Vendas por Semana")  

# Remover as colunas de valores que não existem no dataset test e só foi utilizado para gerar gráficos exploratórios
train_dataset$Venta_uni_hoy <- NULL
train_dataset$Dev_uni_proxima <- NULL

###################### REALIZANDO UM SUBSET DOS DADOS PARA REDUÇÃO DA QUANTIDADE DE REGISTROS ######################
## Esse trecho de código foi utilizando somente na primeira execução, quando ainda estava sendo utilizado o dataset train.csv com 2.5GB de registros.

library(caTools)

#train_dataset_index <- sample.split(train_dataset$Semana, SplitRatio = 0.06)
#train_dataset <- subset(train_dataset, train_dataset_index == TRUE)

#rm(train_dataset_index)

#write.csv(train_dataset, 
#          file = "~/PrevendoDemandaEstoqueBaseVendas/dataset/train_dataset_reduced.csv",
#          sep = ",")

###################### ATRIBUINDO AO DATASET MEDIA POR CLIENTE / CLIENTE + PRODUTO / PRODUTO ###################### 
# Obs.: Também está sendo atribuido os valores de média ao dataset test_dataset para contribuir com a previsão.

# Média por ClienteID
mean_by_clienteID <- train_dataset %>%
  group_by(Cliente_ID) %>%
  summarise(mean_ClienteID = mean(Demanda_uni_equil))

train_dataset <- merge(train_dataset, mean_by_clienteID, by = "Cliente_ID")
test_dataset <- merge(test_dataset, mean_by_clienteID, by = "Cliente_ID")

# Média por CLienteID + ProdutoID
mean_by_clienteID_ProductID <- train_dataset %>%
  group_by(Cliente_ID, Producto_ID) %>%
  summarise(mean_by_clienteID_ProductID = mean(Demanda_uni_equil))

train_dataset <- merge(train_dataset, mean_by_clienteID_ProductID, by = c("Cliente_ID","Producto_ID"))
test_dataset <- merge(test_dataset, mean_by_clienteID_ProductID, by = c("Cliente_ID","Producto_ID"))

# Média por pordutoID
mean_by_ProductID <- train_dataset %>%
  group_by(Producto_ID) %>%
  summarise(mean_by_ProductID = mean(Demanda_uni_equil))

train_dataset <- merge(train_dataset, mean_by_ProductID, by = "Producto_ID")
test_dataset <- merge(test_dataset, mean_by_ProductID, by = "Producto_ID")

###################### VERIFICANDO A CORRELAÇÃO ENTRE VARIÁVEIS ###################### 
# install.packages("corrplot")

library(corrplot)

corDados <- cor(train_dataset)
corrplot(corDados, method = 'color')

###################### SPLIT O DATASET TRAIN.CSV EM TRAIN E AVALIA, PARA SER POSSÍVEL AVALIAR A TAXA DE ACERTO DO MODELO CRIADO ###################### 
train_dataset_index <- sample.split(train_dataset$Semana, SplitRatio = 0.65)
train_dataset_model <- subset(train_dataset, train_dataset_index == TRUE)
evaluate_dataset_model <- subset(train_dataset, train_dataset_index == FALSE)
rm(train_dataset_index)

###################### CRIANDO O MODELO UTILIZANDO GRADIENT BOOSTING ###################### 
#install.packages("gbm")
#install.packages("doParallel")
library(doParallel)
library(gbm)

Sys.time()
gbm_model <- gbm(Demanda_uni_equil ~ . , 
                  data = train_dataset_model,
                  distribution = "gaussian", 
                  n.trees = 100, shrinkage = 0.1,             
                  interaction.depth = 3,
                  cv.folds = 5, 
                  verbose = FALSE, 
                  n.cores = NULL)  
Sys.time()

summary(gbm_model)

## Avaliando o resultado do modelo criado.

result <- data.frame(actual = evaluate_dataset_model$Demanda_uni_equil,
           predict = round(predict(gbm_model, newdata = evaluate_dataset_model)) )

result <- as.vector(result)

result[,"FakeRow"] <- seq(1,nrow(result))

ggplot() +
geom_line(data = result, aes(x = FakeRow, y = actual)) +
  geom_line(data = result, aes(x = FakeRow, y = predict), color = "red") +
  ylab("Demanda Estoque")

# 87% dos dados avalia tiveram uma previsão do valor correto!
result <- mutate(result, resids = predict - actual)
result_f <- result %>% mutate(classf = ifelse(resids == 0, 0, -1))

prop.table(table(result_f$classf))

# APLICANDO O MODELO CRIADO O DATASET TEST CUJO O QUAL NÃO TEMOS COMO AVALIAR A ASSERTIVIDADE, OU SEJA, ESTAMOS APLICANDO ML
# PARA PREVER OS VALORES FUTUROS

predict_final <- round(predict(gbm_model, newdata = test_dataset))

result_final <- data.frame(test_dataset,
                           value_predict = predict_final)