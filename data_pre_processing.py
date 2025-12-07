import numpy as np
import scipy.stats
import sklearn.model_selection
import tensorflow as tf

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    return np.unique(np_array, return_counts=True)[0][np.argmax(np.unique(np_array, return_counts=True)[1])]

def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    overall_window_size = (window_size - 1) * stride + 1
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1
    windows = []
    for i in range(num_windows):
        start_index = i * shift + offset
        this_window = X[start_index : start_index + overall_window_size : stride]
        if flatten is not None:
            this_window = flatten(this_window)
        windows.append(this_window)
    return np.array(windows)

def get_windows_dataset_from_user_list_format(user_datasets, window_size=400, shift=200, stride=1, verbose=0):
    user_dataset_windowed = {}

    for user_id in user_datasets:
        if verbose > 0:
            print(f"Processing {user_id}")
        x = []
        y = []

        # Loop through each trail of each user
        for v,l in user_datasets[user_id]:
            v_windowed = sliding_window_np(v, window_size, shift, stride)
            
            # flatten the window by majority vote (1 value for each window)
            l_flattened = sliding_window_np(l, window_size, shift, stride, flatten=get_mode)
            if len(v_windowed) > 0:
                x.append(v_windowed)
                y.append(l_flattened)
            if verbose > 0:
                print(f"Data: {v_windowed.shape}, Labels: {l_flattened.shape}")

        # combine all trials
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze())
    return user_dataset_windowed

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for user_id in user_datasets_windowed:
        
        v,l = user_datasets_windowed[user_id]
        if user_id in train_users:
            if verbose > 0:
                print(f"{user_id} Train")
            train_x.append(v)
            train_y.append(l)
        elif test_users is None or user_id in test_users:
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v)
            test_y.append(l)
    

    if len(train_x) == 0:
        train_x = np.array([])
        train_y = np.array([])
    else:
        train_x = np.concatenate(train_x)
        train_y = np.concatenate(train_y).squeeze()
    
    if len(test_x) == 0:
        test_x = np.array([])
        test_y = np.array([])
    else:
        test_x = np.concatenate(test_x)
        test_y = np.concatenate(test_y).squeeze()

    return train_x, train_y, test_x, test_y

def get_mean_std_from_user_list_format(user_datasets, train_users):  
    mean_std_data = []
    for u in train_users:
        for data, _ in user_datasets[u]:
            mean_std_data.append(data)
    mean_std_data_combined = np.concatenate(mean_std_data)
    means = np.mean(mean_std_data_combined, axis=0)
    stds = np.std(mean_std_data_combined, axis=0)
    return (means, stds)

def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """

    return ((data - mean) / std)

def apply_label_map(y, label_map):
    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def filter_none_label(X, y):
    valid_mask = np.where(y != None)
    return (np.array(X[valid_mask]), np.array(y[valid_mask], dtype=int))

def pre_process_dataset_composite(user_datasets, label_map, output_shape, train_users, test_users, window_size, shift, normalise_dataset=True, validation_split_proportion=0.2, verbose=0):
    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    # Step 2
    train_x, train_y, test_x, test_y = combine_windowed_dataset(user_datasets_windowed, train_users, test_users)

    # Step 3
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)
        train_x = normalise(train_x, means, stds)
        if len(test_users) > 0:
            test_x = normalise(test_x, means, stds)
        else:
            test_x = np.empty((0,0))

    # Step 4
    train_y_mapped = apply_label_map(train_y, label_map)
    test_y_mapped = apply_label_map(test_y, label_map)

    train_x, train_y_mapped = filter_none_label(train_x, train_y_mapped)
    test_x, test_y_mapped = filter_none_label(test_x, test_y_mapped)

    if verbose > 0:
        print("Test")
        print(np.unique(test_y, return_counts=True))
        print(np.unique(test_y_mapped, return_counts=True))
        print("-----------------")

        print("Train")
        print(np.unique(train_y, return_counts=True))
        print(np.unique(train_y_mapped, return_counts=True))
        print("-----------------")

    # Step 5
    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape)
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape)

    r = np.random.randint(len(train_y_mapped))
    assert train_y_one_hot[r].argmax() == train_y_mapped[r]
    if len(test_users) > 0:
        r = np.random.randint(len(test_y_mapped))
        assert test_y_one_hot[r].argmax() == test_y_mapped[r]

    # Step 6
    if validation_split_proportion is not None and validation_split_proportion > 0:
        train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x, train_y_one_hot, test_size=validation_split_proportion, random_state=42)
    else:
        train_x_split = train_x
        train_y_split = train_y_one_hot
        val_x_split = None
        val_y_split = None
        

    if verbose > 0:
        print("Training data shape:", train_x_split.shape)
        print("Validation data shape:", val_x_split.shape if val_x_split is not None else "None")
        print("Testing data shape:", test_x.shape)

    np_train = (train_x_split, train_y_split)
    np_val = (val_x_split, val_y_split) if val_x_split is not None else None
    np_test = (test_x, test_y_one_hot)

    return (np_train, np_val, np_test)

def pre_process_dataset_composite_in_user_format(user_datasets, label_map, output_shape, train_users, window_size, shift, normalise_dataset=True, verbose=0):
    # Preparation for step 2
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)

    # Step 1
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    
    user_datasets_processed = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Step 2
        labels_mapped = apply_label_map(labels, label_map)
        data_filtered, labels_filtered = filter_none_label(data, labels_mapped)

        # Step 3
        labels_one_hot = tf.keras.utils.to_categorical(labels_filtered, num_classes=output_shape)

        # random check
        r = np.random.randint(len(labels_filtered))
        assert labels_one_hot[r].argmax() == labels_filtered[r]

        # Step 4
        if normalise_dataset:
            data_filtered = normalise(data_filtered, means, stds)

        user_datasets_processed[user] = (data_filtered, labels_one_hot)

        if verbose > 0:
            print("Data shape of user", user, ":", data_filtered.shape)
    
    return user_datasets_processed

def add_user_id_to_windowed_dataset(user_datasets_windowed, encode_user_id=True, as_feature=False, as_label=True, verbose=0):
    # Create the mapping from user_id to integers
    if encode_user_id:
        all_users = sorted(list(user_datasets_windowed.keys()))
        user_id_encoder = dict([(u, i) for i, u in enumerate(all_users)])
    else:
        user_id_encoder = None

    # if none of the options are enabled, return the input
    if not as_feature and not as_label:
        return user_datasets_windowed, user_id_encoder

    user_datasets_modified = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Get the encoded user_id
        if encode_user_id:
            user_id = user_id_encoder[user]
        else:
            user_id = user

        # Add user_id as an extra feature
        if as_feature:
            user_feature = np.expand_dims(np.full(data.shape[:-1], user_id), axis=-1)
            data_modified = np.append(data, user_feature, axis=-1)
        else:
            data_modified = data
        
        # Add user_id as an extra label
        if as_label:
            user_labels = np.expand_dims(np.full(labels.shape[:-1], user_id), axis=-1)
            labels_modified = np.append(labels, user_labels, axis=-1)
        else:
            labels_modified = labels

        if verbose > 0:
            print(f"User {user}: id {repr(user)} -> {repr(user_id)}, data shape {data.shape} -> {data_modified.shape}, labels shape {labels.shape} -> {labels_modified.shape}")

        user_datasets_modified[user] = (data_modified, labels_modified)
    
    return user_datasets_modified, user_id_encoder

def make_batches_reshape(data, batch_size):
    max_len = (data.shape[0]) // batch_size * batch_size
    return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

def np_random_shuffle_index(length):
    """
    Get a list of randomly shuffled indices
    """
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_batched_dataset_generator(data, batch_size):
    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

    # return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))
