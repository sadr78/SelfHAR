import tensorflow as tf

def create_1d_conv_core_model(input_shape, model_name="base_model", use_standard_max_pooling=False):
    inputs = tf.keras.Input(shape=input_shape, name='input')
    x = inputs
    x = tf.keras.layers.Conv1D(
            32, 24,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
            64, 16,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    if use_standard_max_pooling:
        x = tf.keras.layers.MaxPool1D(pool_size=x.shape[1], padding='valid', data_format='channels_last', name='max_pooling1d')(x)
        x = tf.keras.layers.Reshape([x.shape[-1]], name='reshape_squeeze')(x)
    else:
        x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    return tf.keras.Model(inputs, x, name=model_name)


def extract_core_model(composite_model):
    return composite_model.layers[1]

def extract_har_model(multitask_model, optimizer, output_index=-1, model_name="har"):
    model = tf.keras.Model(inputs=multitask_model.inputs, outputs=multitask_model.outputs[output_index], name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model

def set_freeze_layers(model, num_freeze_layer_index=None):
    if num_freeze_layer_index is None:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:num_freeze_layer_index]:
            layer.trainable = False
        for layer in model.layers[num_freeze_layer_index:]:
            layer.trainable = True


def attach_full_har_classification_head(core_model, output_shape, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), num_units=1024, model_name="HAR"):
    inputs = tf.keras.Input(shape=core_model.inputs[0].shape[1:], name='input')
    intermediate_x = core_model(inputs)

    x = tf.keras.layers.Dense(num_units, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model

def attach_linear_classification_head(core_model, output_shape, optimizer=tf.keras.optimizers.SGD(learning_rate=0.03), model_name="Linear"):
    inputs = tf.keras.Input(shape=core_model.inputs[0].shape[1:], name='input')
    intermediate_x = core_model(inputs)

    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(intermediate_x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def attach_multitask_transform_head(core_model, output_tasks, optimizer, with_har_head=False, har_output_shape=None, num_units_har=1024, model_name="multitask_transform"):
    """
    Note: core_model is also modified after training this model (i.e. the weights are updated)
    """
    inputs = tf.keras.Input(shape=core_model.inputs[0].shape[1:], name='input')
    intermediate_x = core_model(inputs)
    outputs = []
    losses = [tf.keras.losses.BinaryCrossentropy() for _ in output_tasks]
    for task in output_tasks:
        x = tf.keras.layers.Dense(256, activation='relu')(intermediate_x)
        pred = tf.keras.layers.Dense(1, activation='sigmoid', name=task)(x)
        outputs.append(pred)


    if with_har_head:
        x = tf.keras.layers.Dense(num_units_har, activation='relu')(intermediate_x)
        x = tf.keras.layers.Dense(har_output_shape)(x)
        har_pred = tf.keras.layers.Softmax(name='har')(x)

        outputs.append(har_pred)
        losses.append(tf.keras.losses.CategoricalCrossentropy())

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['accuracy']*len(model.outputs)
    )
    
    return model
