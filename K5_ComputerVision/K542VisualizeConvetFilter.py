import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    # The call to `gradients` returns a list of tensors (of size 1 in this case)
    # hence we only keep the first element -- which is a tensor.
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        # Compute the loss value and gradient value
        loss_value, grads_value = iterate([input_img_data])
        # Here we adjust the input image in the direction that maximizes the loss
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)

model = VGG16(weights='imagenet',include_top=False)

layer_name = 'block4_conv1'
filter_index = 1
image1= generate_pattern(layer_name, filter_index)
plt.imshow(image1)
plt.show()

layer_name = 'block4_conv1'
size = 64
margin = 5
results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3),dtype='uint8')
for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start:horizontal_end,vertical_start:vertical_end, :] = filter_img
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()
print('end')