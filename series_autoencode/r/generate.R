##################################################
# 生成序列数据
##################################################
require(ggplot2)
require(reshape2)
require(plyr)


# 演示效果
x <- seq(-1.5*pi, 1.5*pi, by = 0.3)
d1 <- data.frame(x = x, y = sin(0.5*x), type = 'sin(0.5x)')
d2 <- data.frame(x = x, y = tanh(x), type = 'tanh(x)')
d3 <- data.frame(x = x, y = sin(2*x), type = 'sin(2x)')
d <- rbind(d1,d2,d3)
d$y_noise <- d$y + rnorm(nrow(d),sd=0.5)
p <- ggplot(d,aes(x, y_noise, color = type))
p <- p + geom_line(size = 1)
p

# 生成数据
set.seed(34354)
SAMPLE_SIZE <- 1000
UNIT <- 1.5
STEP <- 0.1
NOISE_SD <- 0.5
single_x <- seq(-UNIT*pi, UNIT*pi, by = 0.25)

my_data <- data.frame()
for(i in 1:SAMPLE_SIZE) {
  x_count <- length(single_x)
  index <- 1:x_count
  d1 <- data.frame(x = single_x,
                   x_index = index,
                   y = sin(0.5*single_x) + rnorm(x_count,sd=NOISE_SD),
                   type = 'sin(0.5x)')
    
  d2 <- data.frame(x = single_x, 
                   x_index = index,
                   y = tanh(single_x) + rnorm(x_count,sd=NOISE_SD), 
                   type = 'tanh(x)')
  
  d3 <- data.frame(x = single_x,
                   x_index = index,
                   y = sin(2*single_x) + rnorm(x_count,sd=NOISE_SD),
                   type = 'sin(2x)')
  
  d <- rbind(d1,d2,d3)
  d$id <- i
  
  my_data <- rbind(my_data, d)
  
  
  if(i %% 10 == 0)
    print(sprintf("Complete %d", i))
}



#　抽样，确保边变长
set.seed(3)
n <- nrow(my_data)
sub_my_data <- my_data[sample(n, n*0.5),]

sample_id <- sample(SAMPLE_SIZE, 1)
p <- ggplot(sub_my_data[sub_my_data$id==sample_id,],
            aes(x,y, color = type))
p <- p + geom_line(size=1.5)
p

# 数据打横
my_data_h <- dcast(
  sub_my_data,id+type~x_index, 
  value.var = 'y'
)
dim(my_data_h)

# 格式化
col_num <- ncol(my_data_h)
my_data_series <- ddply(
  my_data_h,
  .(id,type),
  function(x) {
     features <- x[1,3:col_num]
     non_na <- features[1,!is.na(features[1,])]
     padding <- c(non_na, rep(0, col_num - ncol(non_na)))
     c(series = paste(padding, collapse = " "), series_length = ncol(non_na))
  }
)

# 写数据
write.csv(
  my_data_series, 
  file = 'data/sd_0.5_n_1000.csv',
  quote = F,
  row.names = F 
)
 
write.csv(
  sub_my_data, 
  file = 'data/sd_0.5_n_1000_v.csv',
  quote = F,
  row.names = F
)

