###########################################################################
# 使用随机深林模型,评估不同auto encode方法对特征提取的效果
###########################################################################
require(reshape2)
require(randomForest)
require(plyr)

emb_ae_tag <- read.csv("tanh_sd_0.5_n_1000_ae_tag.csv")
emb_ae_tag$id <- sprintf("%s-%s", emb_ae_tag$type, emb_ae_tag$id)

emb_ae <- read.csv("c_tanh_sd_1_n_1000_ae.csv")
emb_ae$id <- sprintf("%s-%s", emb_ae$type, emb_ae$id)

original <- read.csv("tanh_sd_1_n_1000_v.csv")
original_h <- dcast(
  original,
  id+type~x_index, 
  value.var = 'y'
)
original_h$id <- sprintf("%s-%s", original_h$type, original_h$id)
sub_original_h <- original_h[original_h$id %in% emb_ae$id, ]
names(sub_original_h) <- c("id","type", paste("e", 1:(ncol(sub_original_h)-2), sep="_"))


# 生成k fold交叉验证的索引
# n 总样本量 
# k 交叉数量
k_fold_index <- function(n, k = 5) {
  index <- rep(1:k, length.out = n)
  sample(index, length(index), replace = FALSE)
}

set.seed(3243)
K <- 5
N <- nrow(sub_original_h)
train_validate_index <- k_fold_index(N, k = K)
result <- data.frame()
for(i in 1:K) {
  train_index <-  train_validate_index != i
  validate_index <- train_validate_index == i
  
  # 训练原始数据
  train_data <- sub_original_h[train_index, ]
  train_data$id <- NULL
  validate_data <- sub_original_h[validate_index,]
  
  model <- randomForest(type~.,train_data)
  pred <- predict(model, validate_data)
  cm <- table(pred, validate_data$type)
  
  
  
  error <- sum(pred != validate_data$type)/nrow(validate_data)
  current_result <- data.frame(r=i, type = 'original', error = error)
  print(current_result)
  result <- rbind(result, current_result)
  
  # 训练自我编码
  train_data <- emb_ae[train_index, ]
  train_data$id <- NULL
  validate_data <- emb_ae[validate_index,]
  
  model <- randomForest(type~.,train_data)
  pred <- predict(model, validate_data)
  
  cm2 <- table(pred, validate_data$type)
  
  error <- sum(pred != validate_data$type)/nrow(validate_data)
  current_result <- data.frame(r=i, type = 'ae', error = error)
  print(current_result)
  result <- rbind(result, current_result)
  
  # 训练标签自我编码
  train_data <- emb_ae_tag[train_index, ]
  train_data$id <- NULL
  validate_data <- emb_ae_tag[validate_index,]
  
  model <- randomForest(type~.,train_data)
  pred <- predict(model, validate_data)
  cm3 <- table(pred, validate_data$type)
  
  error <- sum(pred != validate_data$type)/nrow(validate_data)
  
  
  
  current_result <- data.frame(r=i, type = 'ae tag', error = error)
  print(current_result)
  result <- rbind(result, current_result)
}

ddply(
  result,
  .(type),
  function(x) c(error = mean(x$error))
)


feature_index <- grep("e_\\d+",names(sub_original_h))
pca <- prcomp(sub_original_h[,feature_index])
plot(pca)

data_with_tag <- as.data.frame(pca$x)
data_with_tag$tag <- sub_original_h$type

p <- ggplot(data_with_tag, aes(x=PC1,y=PC2, color=tag))
p <- p + geom_point()
p

