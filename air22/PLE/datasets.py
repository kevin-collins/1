import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def build_tensor_dataset(x, y):
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


def data_preparation(seed=0, test_size=0.5):
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    train_df = pd.read_csv('datasets/census_income/census-income.data.gz', delimiter=',', header=None, index_col=None,
                           names=column_names)
    test_df = pd.read_csv('datasets/census_income/census-income.test.gz', delimiter=',', header=None, index_col=None,
                          names=column_names)

    label_columns = ['income_50k', 'marital_stat']

    train_labels = train_df[label_columns].copy()
    test_labels = test_df[label_columns].copy()
    train_labels.loc[:, 'income_50k'] = train_labels.loc[:, 'income_50k'].apply(lambda x: int(x == ' 50000+.'))
    train_labels.loc[:, 'marital_stat'] = train_labels.loc[:, 'marital_stat'].apply(lambda x: int(x == ' Never married'))
    test_labels.loc[:, 'income_50k'] = test_labels.loc[:, 'income_50k'].apply(lambda x: int(x == ' 50000+.'))
    test_labels.loc[:, 'marital_stat'] = test_labels.loc[:, 'marital_stat'].apply(lambda x: int(x == ' Never married'))

    # One-hot encoding categorical columns
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_transformed = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    test_transformed = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)

    # Filling the missing column in the other set
    test_transformed['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0

    train_data = train_transformed.to_numpy()
    train_label = train_labels.to_numpy()

    # Split the test datasets into (1-test_size):(test_size) validation to test according to the paper
    valid_data, test_data, valid_label, test_label = train_test_split(test_transformed.to_numpy(), test_labels.to_numpy(),
                                                                      test_size=test_size, random_state=seed)

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(valid_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    return train_data, train_label, valid_data, valid_label, test_data, test_label
