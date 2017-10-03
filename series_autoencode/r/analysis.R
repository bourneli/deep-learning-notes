require(ggplot2)

# encode_file <- 'tanh_sd_1_n_1000_ae_tag.csv'
encode_file <- 'data/c_sd_0.5_n_1000_ae.csv'
# encode_file <- 'data/h_sd_0.1_n_1000_ae.csv'
embedding <- read.csv(encode_file)

feature_index <- grep("e_\\d+",names(embedding))
pca <- prcomp(embedding[,feature_index])
plot(pca)

data_with_tag <- as.data.frame(pca$x)
data_with_tag$tag <- embedding$type

p <- ggplot(data_with_tag, aes(x=PC1,y=PC2, color=tag))
p <- p + geom_point()
p

p <- ggplot(data_with_tag, aes(x=PC1,y=PC2, color=tag))
p <- p + geom_density2d()  
p
