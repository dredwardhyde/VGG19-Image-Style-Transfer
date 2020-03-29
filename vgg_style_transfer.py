import tensorflow as tf
import numpy as np
import PIL.Image
import time

content_path = './results/man_source.jpg'
style_path = './results/spaghetti_source.jpg'


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


content_image = load_img(content_path)
style_image = load_img(style_path)

content_layers = ['block5_conv2',
                  'block5_conv3',
                  'block5_conv4']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def gram_matrix(input_tensor):
    input_shape = tf.shape(input_tensor)
    # Get the number of feature channels for the input tensor,
    # which is assumed to be from a convolutional layer with 4-dim.
    num_channels = int(input_shape[3])
    # Reshape the tensor so it is a 2-dim matrix. This essentially
    # flattens the contents of each feature-channel.
    matrix = tf.reshape(input_tensor, shape=[-1, num_channels])
    # Calculate the Gram-matrix as the matrix-product of
    # the 2-dim matrix with itself. This calculates the
    # dot-products of all combinations of the feature-channels.
    gram = tf.matmul(tf.transpose(matrix), matrix)
    # Average that outer product over all locations
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return gram / num_locations


vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
model = tf.keras.Model([vgg.input], outputs)
model.trainable = False


def style_transfer(inputs):
    inputs = inputs * 255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = model(preprocessed_input)
    style_outputs, content_outputs = (outputs[:num_style_layers], outputs[num_style_layers:])
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    content_dict = {content_name: value for content_name, value in zip(content_layers, content_outputs)}
    style_dict = {style_name: value for style_name, value in zip(style_layers, style_outputs)}
    return {'content': content_dict, 'style': style_dict}


style_targets = style_transfer(style_image)['style']
content_targets = style_transfer(content_image)['content']
image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-3)
style_weight = 0.5
content_weight = 10000
total_variation_weight = 10


def train_step(image):
    with tf.GradientTape() as tape:
        outputs = style_transfer(image)
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n(
            [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers
        content_loss = tf.add_n(
            [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        loss += total_variation_weight * tf.image.total_variation(image)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


epochs = 100
steps_per_epoch = 100
start = time.time()
step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='')
    tensor_to_image(image).save('result_' + str(n) + '.png')
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))
