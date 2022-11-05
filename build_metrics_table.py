from typing import List, Dict
import os
import numpy as np
import pandas as pd

from utils.table_tools import read_gt_table, read_colorchecker_table
from utils.metrics import rmse, calculate_delta_e_between_tables


# Remove .DS_Store which used in MACOS file system
def remove_dsstore(names_list: List[str]) -> List[str]:
    return [
        name
        for name in names_list
        if '.DS_Store' not in name
    ]


def filtered_listdir(_path: str) -> List[str]:
    return remove_dsstore(os.listdir(_path))


data_folder = './data/images_dataset/'
images_set_folders = os.path.join(data_folder, 'images/')
checker_data_folders = os.path.join(data_folder, 'checker_results/')
experiments_results_subfolders = filtered_listdir(checker_data_folders)

images_names = sorted(
    filtered_listdir(
        os.path.join(images_set_folders, experiments_results_subfolders[0])
    ),
    key=lambda x: int(os.path.splitext(x)[0].split('DSC')[-1])
)

table_data = {
    'Experiment No': [],
    'Image name': [],
    'Collection name': [],
    'RMSE': [],
    'Delta E': []
}

gt_rgbs, gt_labs = read_gt_table('./gt_colors.csv')

for exp_idx, image_name in enumerate(images_names):
    base_name = os.path.splitext(image_name)[0]
    for set_name in sorted(filtered_listdir(checker_data_folders)):
        table_path = os.path.join(
            checker_data_folders, set_name, base_name + '-MCC-values.csv'
        )

        sample_rgbs, sample_labs = read_colorchecker_table(table_path)

        sample_rmse = rmse(sample_rgbs, gt_rgbs)
        sample_delta_e = calculate_delta_e_between_tables(sample_labs, gt_labs)

        table_data['Experiment No'].append(exp_idx + 1)
        table_data['Image name'].append(base_name)
        table_data['Collection name'].append(set_name.replace('calibrated', 'corrected'))
        table_data['RMSE'].append(sample_rmse)
        table_data['Delta E'].append(sample_delta_e)

statistic_table_data = {
    'Collection name': [],
    'RMSE mean': [],
    'RMSE std': [],
    'Delta E mean': [],
    'Delta E std': []
}

calib_rmse: Dict[str, float] = dict()
calib_delta_e: Dict[str, float] = dict()

for exp_idx, image_name in enumerate(images_names):
    set_name = 'sony_calibrated'
    base_name = os.path.splitext(image_name)[0]

    table_path = os.path.join(
        checker_data_folders, set_name, base_name + '-MCC-values.csv'
    )

    sample_rgbs, sample_labs = read_colorchecker_table(table_path)

    sample_rmse = rmse(sample_rgbs, gt_rgbs)
    sample_delta_e = calculate_delta_e_between_tables(sample_labs, gt_labs)

    calib_rmse[base_name] = sample_rmse
    calib_delta_e[base_name] = sample_delta_e

for set_name in sorted(filtered_listdir(checker_data_folders)):
    if set_name == 'sony' or set_name == 'sony_calibrated':
        continue

    with_calib_diff_rmse = []
    with_calib_diff_delta_e = []

    for exp_idx, image_name in enumerate(images_names):
        base_name = os.path.splitext(image_name)[0]
        table_path = os.path.join(
            checker_data_folders, set_name, base_name + '-MCC-values.csv'
        )

        sample_rgbs, sample_labs = read_colorchecker_table(table_path)

        sample_rmse = rmse(sample_rgbs, gt_rgbs)
        sample_delta_e = calculate_delta_e_between_tables(sample_labs, gt_labs)

        with_calib_diff_rmse.append(
            np.abs(calib_rmse[base_name] - sample_rmse)
        )
        with_calib_diff_delta_e.append(
            np.abs(calib_delta_e[base_name] - sample_delta_e)
        )

    statistic_table_data['Collection name'].append(
        set_name.replace('calibrated', 'corrected')
    )
    statistic_table_data['RMSE mean'].append(np.mean(with_calib_diff_rmse))
    statistic_table_data['RMSE std'].append(np.std(with_calib_diff_rmse))
    statistic_table_data['Delta E mean'].append(np.mean(with_calib_diff_delta_e))
    statistic_table_data['Delta E std'].append(np.std(with_calib_diff_delta_e))

results_table = pd.DataFrame(data=table_data)
results_table.to_csv('./data/experiment_results.csv', index=False, sep=',')

stats_table = pd.DataFrame(data=statistic_table_data)
stats_table.to_csv('./data/experiment_stats.csv', index=False, sep=',')
