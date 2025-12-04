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

# ===================================================================
#   Transformer Core Model + Teacher/Student Transformer Models
#   (Patch Added)
# ===================================================================

import tensorflow as tf


def create_transformer_core_model(
    input_shape,
    model_name="transformer_core",
    d_model=128,
    num_heads=4,
    dropout_rate=0.1,
):
    """
    Lightweight Transformer encoder for time-series HAR.
    Produces a feature vector similar to create_1d_conv_core_model().
    """

    inputs = tf.keras.Input(shape=input_shape, name="input")

    # Project input → d_model dim
    x = tf.keras.layers.Dense(d_model, name="projection_dense")(inputs)

    # Positional embedding
    seq_len = input_shape[0]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=d_model, name="pos_embedding",
    )(positions)
    x = x + pos_embedding

    # Transformer encoder layers
    for i in range(num_layers):
        # MHA block
        attn_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")(x)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"mha_{i}",
        )(attn_in, attn_in)
        x = tf.keras.layers.Add(name=f"attn_residual_{i}")([x, attn_out])

        # FFN block
        ffn_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{i}")(x)
        ffn = tf.keras.layers.Dense(dff, activation="relu", name=f"ffn_dense1_{i}")(ffn_in)
        ffn = tf.keras.layers.Dropout(dropout_rate, name=f"ffn_dropout_{i}")(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f"ffn_dense2_{i}")(ffn)
        x = tf.keras.layers.Add(name=f"ffn_residual_{i}")([x, ffn])

    # Global average pooling for final feature vector
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    core_model = tf.keras.Model(inputs=inputs, outputs=x, name=model_name)
    return core_model



# ===============================================================
#   Teacher Transformer  /  Student Transformer Builders
# ===============================================================

def build_teacher_student_transformer_models(
    input_shape,
    num_classes,
    optimizer_teacher=None,
    optimizer_student=None,
    num_units_head=1024,
):
    """
    Builds:
      Teacher_Transformer
      Student_Transformer
    """

    if optimizer_teacher is None:
        optimizer_teacher = tf.keras.optimizers.Adam(3e-4)

    if optimizer_student is None:
        optimizer_student = tf.keras.optimizers.Adam(3e-4)

    # ---- Teacher core ----
    teacher_core = create_transformer_core_model(
        input_shape=input_shape,
        model_name="teacher_transformer_core",
    )

    # ---- Student core ----
    student_core = create_transformer_core_model(
        input_shape=input_shape,
        model_name="student_transformer_core",
    )

    # --- classification heads (existing) ---
    teacher_model = attach_full_har_classification_head(
        core_model=teacher_core,
        output_shape=num_classes,
        optimizer=optimizer_teacher,
        num_units=num_units_head,
        model_name="Teacher_Transformer",
    )

    student_model = attach_full_har_classification_head(
        core_model=student_core,
        output_shape=num_classes,
        optimizer=optimizer_student,
        num_units=num_units_head,
        model_name="Student_Transformer",
    )

    return teacher_model, student_model


# ===================================================================
#   Transformer Core Model + Teacher/Student Transformer Models
#   (Patch Added)
# ===================================================================

import tensorflow as tf


def create_transformer_core_model(
    input_shape,
    model_name="transformer_core",
    d_model=128,
    num_heads=4,
    dff=256,
    num_layers=2,
    dropout_rate=0.1,
):
    """
    Lightweight Transformer encoder for time-series HAR.
    Produces a feature vector similar to create_1d_conv_core_model().
    """

    inputs = tf.keras.Input(shape=input_shape, name="input")

    # Project input → d_model dim
    x = tf.keras.layers.Dense(d_model, name="projection_dense")(inputs)

    # Positional embedding
    seq_len = input_shape[0]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=d_model, name="pos_embedding",
    )(positions)
    x = x + pos_embedding

    # Transformer encoder layers
    for i in range(num_layers):
        # MHA block
        attn_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")(x)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"mha_{i}",
        )(attn_in, attn_in)
        x = tf.keras.layers.Add(name=f"attn_residual_{i}")([x, attn_out])

        # FFN block
        ffn_in = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{i}")(x)
        ffn = tf.keras.layers.Dense(dff, activation="relu", name=f"ffn_dense1_{i}")(ffn_in)
        ffn = tf.keras.layers.Dropout(dropout_rate, name=f"ffn_dropout_{i}")(ffn)
        ffn = tf.keras.layers.Dense(d_model, name=f"ffn_dense2_{i}")(ffn)
        x = tf.keras.layers.Add(name=f"ffn_residual_{i}")([x, ffn])

    # Global average pooling for final feature vector
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    core_model = tf.keras.Model(inputs=inputs, outputs=x, name=model_name)
    return core_model



# ===============================================================
#   Teacher Transformer  /  Student Transformer Builders
# ===============================================================

def build_teacher_student_transformer_models(
    input_shape,
    num_classes,
    optimizer_teacher=None,
    optimizer_student=None,
    num_units_head=1024,
):
    """
    Builds:
      Teacher_Transformer
      Student_Transformer
    """

    if optimizer_teacher is None:
        optimizer_teacher = tf.keras.optimizers.Adam(3e-4)

    if optimizer_student is None:
        optimizer_student = tf.keras.optimizers.Adam(3e-4)

    # ---- Teacher core ----
    teacher_core = create_transformer_core_model(
        input_shape=input_shape,
        model_name="teacher_transformer_core",
    )

    # ---- Student core ----
    student_core = create_transformer_core_model(
        input_shape=input_shape,
        model_name="student_transformer_core",
    )

    # --- classification heads (existing) ---
    teacher_model = attach_full_har_classification_head(
        core_model=teacher_core,
        output_shape=num_classes,
        optimizer=optimizer_teacher,
        num_units=num_units_head,
        model_name="Teacher_Transformer",
    )

    student_model = attach_full_har_classification_head(
        core_model=student_core,
        output_shape=num_classes,
        optimizer=optimizer_student,
        num_units=num_units_head,
        model_name="Student_Transformer",
    )

    return teacher_model, student_model



# ===================================================================
# PATCH 1 — Transformer Core + Teacher & Student Transformer
# ===================================================================

import tensorflow as tf


def create_transformer_core_model(
    input_shape,
    model_name="transformer_core",
    d_model=128,
    num_heads=4,
    dff=256,
    num_layers=2,
    dropout_rate=0.1,
):
    inputs = tf.keras.Input(shape=input_shape, name="input")

    x = tf.keras.layers.Dense(d_model)(inputs)

    seq_len = input_shape[0]
    pos = tf.range(0, seq_len)
    pos_embed = tf.keras.layers.Embedding(seq_len, d_model)(pos)
    x = x + pos_embed

    for i in range(num_layers):
        ln1 = tf.keras.layers.LayerNormalization()(x)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(ln1, ln1)
        x = tf.keras.layers.Add()([x, attn])

        ln2 = tf.keras.layers.LayerNormalization()(x)
        ffn = tf.keras.layers.Dense(dff, activation="relu")(ln2)
        ffn = tf.keras.layers.Dense(d_model)(ffn)
        x = tf.keras.layers.Add()([x, ffn])

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return tf.keras.Model(inputs, x, name=model_name)


def build_teacher_student_transformer_models(
    input_shape,
    num_classes,
    optimizer_teacher=None,
    optimizer_student=None,
    num_units_head=1024,
):

    if optimizer_teacher is None:
        optimizer_teacher = tf.keras.optimizers.Adam(3e-4)
    if optimizer_student is None:
        optimizer_student = tf.keras.optimizers.Adam(3e-4)

    teacher_core = create_transformer_core_model(input_shape)
    student_core = create_transformer_core_model(input_shape)

    teacher = attach_full_har_classification_head(
        teacher_core, num_classes, optimizer_teacher, num_units_head, "Teacher_Transformer"
    )

    student = attach_full_har_classification_head(
        student_core, num_classes, optimizer_student, num_units_head, "Student_Transformer"
    )

    return teacher, student

