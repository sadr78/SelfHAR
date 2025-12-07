import glob
import re
import os
import pandas as pd
import numpy as np

def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    
    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    # print(data_folder_path)
    har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv')) # "<PATH_TO_HHAR_DATASET>/Phones_accelerometer.csv"
    har_dataset.dropna(how = "any", inplace = True)
    har_dataset = har_dataset[["x", "y", "z", "gt","User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data,labels)]
    
    return user_datasets