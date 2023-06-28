# install.packages("tensorflow")
# install.packages("purr")
# install.packages("mapview")
# install.packages("terra")
library(keras)
library(tensorflow)
library(tfdatasets)
library(purrr)
library(ggplot2)
library(rsample)
library(reticulate)
library(mapview)
library(terra)

setwd("C:/Users/Chris/Desktop/Lectures/Semester_2/Advanced_programming/airfield_train")

r <- terra::rast("./airfield_aoi.tif")
viewRGB(as(r, "Raster"))

# Data preprocessing
data <- rbind(
  data.frame(
    img = list.files('./instance/true', full.names = T),
    labels = 1
  ),
  data.frame(
    img = list.files('./instance/false', full.names = T), #. is directory
    labels = 0
  )
)

#samples
n_true <- length(which(data$labels == 1))
n_false <- length(which(data$labels == 0))

#stratified split
data <- initial_split(data, prop = 0.75, strata = "labels")

#check whether the stratified sampling worked
length(which(training(data)$labels == 1))
length(which(training(data)$labels == 0))
length(which(testing(data)$labels == 1))
length(which(testing(data)$labels == 0))


#build tensor slices for each field din training (data) and populate it
#with values
train_ds <- tensor_slices_dataset(training(data)) #this happens in the tensor flow backend
class(train_ds)

#function to extract contents of tensorslice dataset
slices2list <- function(x){
  iterate(as_iterator(x))
}
train_ds_list <- slices2list(train_ds)
# View(train_ds_list)
train_ds_list[[1]]

input_shape <-  c(64,64,3)

train_ds <- dataset_map(train_ds, function(x){
  list_modify(x,img = tf$image$decode_jpeg(tf$io$read_file(x$img)))
})

#settle on a universal data-type
train_ds <- dataset_map(train_ds, function(x){
  list_modify(x, img = tf$image$convert_image_dtype(x$img, dtype = tf$float32))
})

#resize
train_ds <- dataset_map(train_ds, function(x){
  list_modify(
    x,
    img = tf$image$resize(x$img, size = shape(input_shape[1],
                                               input_shape[2])))
})

#shuffling
train_ds <- dataset_shuffle(train_ds, buffer_size = 1280)

#create batches
train_ds <- dataset_batch(train_ds, 10)

#unname dataset, remove all names and adresses
train_ds <- dataset_map(train_ds,unname)

##Validation
test_ds <- tensor_slices_dataset(testing(data))

#function to extract contents of tensorslice dataset
slices2list <- function(x){
  iterate(as_iterator(x))
}
test_ds_list <- slices2list(test_ds)
# View(train_ds_list)
test_ds_list[[1]]

input_shape <-  c(64,64,3)

test_ds <- dataset_map(test_ds, function(x){
  list_modify(x,img = tf$image$decode_jpeg(tf$io$read_file(x$img)))
})

#settle on a universal data-type
test_ds <- dataset_map(test_ds, function(x){
  list_modify(x, img = tf$image$convert_image_dtype(x$img, dtype = tf$float32))
})

#resize
test_ds <- dataset_map(test_ds, function(x){
  list_modify(
    x,
    img = tf$image$resize(x$img, size = shape(input_shape[1],
                                              input_shape[2])))
})

# #shuffling
# test_ds <- dataset_shuffle(test_ds, buffer_size = 1280)

#create batches
test_ds <- dataset_batch(test_ds, 10)

#unname dataset, remove all names and adresses
test_ds <- dataset_map(test_ds,unname)

#network design
model <- keras_model_sequential()

n_filter <- 32
# reg_param <- 0.0009
model%>%
  layer_conv_2d(filters = n_filter, kernel_size = 3, activation = 'relu', input_shape = input_shape )%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = n_filter*2, kernel_size = 3, activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = n_filter*4, kernel_size = 3, activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_conv_2d(filters = n_filter*4, kernel_size = 3, activation = 'relu')%>%
  layer_max_pooling_2d(pool_size = c(2,2))%>%
  layer_flatten()%>%
  layer_dense(units = 256, activation = 'relu')%>%
  layer_dense(unit = 1, activation = 'sigmoid')

summary(model)

#learning design
#define a learning design on the necessary parameters and function we want to test
#optimizer = msprop

model%>%compile(loss = 'binary_crossentropy',
optimizer = optimizer_rmsprop(lr = 0.00005),
metrics = 'accuracy'
)

# learning
train_fit <- fit(
  model, train_ds,
  epochs = 101, validation_data = test_ds,
  callbacks = list(
    callback_early_stopping(monitor = "val_accuracy", patience = 5)
  )
)
