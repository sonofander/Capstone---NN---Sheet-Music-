model = Sequential()
model.add(Convolution2D(20, 4, 4, input_shape=(300, 300, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Added Round 2
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Convolution2D(16, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))



#
# if conditional({{choice(['two', 'three'])}}) == 'three':
#     model.add(Convolution2D({{choice([10, 12, 16, 20, 24, 30])}}, {{choice([3, 4])}}, {{choice([3,4])}}))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size={{choice((2,2),(3,3),(4,4))}}))
#
# if conditional({{choice(['three', 'four'])}}) == 'four':
#     model.add(Convolution2D({{choice([10, 12, 16, 20, 24, 30])}}, {{choice([3, 4])}}, {{choice([3,4])}}))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size={{choice((2,2),(3,3),(4,4))}}))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Convolution2D({{choice([10, 12, 16])}}, 4, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Convolution2D({{choice([10, 12, 16])}}, 4, 3)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
