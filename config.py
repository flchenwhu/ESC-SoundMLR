ESC_50 = False
US8K = False
ESC51 = True




if ESC_50:
    class_numbers = 50
elif ESC51:
    class_numbers = 51
else:
    class_numbers = 10


if ESC_50:
    lr = 5e-4
    folds = 5
    test_fold = [1]
    train_folds = list(i for i in range(1, 6) if i != test_fold[0])
elif ESC51:
    lr = 5e-4
    folds = 5
    test_fold = [1]
    train_folds = list(i for i in range(1, 6) if i != test_fold[0])
else:
    lr = 1e-4  # for US8K
    fold = 10
    test_fold = [1]
    train_folds = list(i for i in range(1, 11) if i != test_fold[0])



temperature = 0.05
alpha = 0
beta = 0.5


freq_masks = 2
time_masks = 1
freq_masks_width = 32
time_masks_width = 32

batch_size = 64
warm_epochs = 10
gamma = 0.98
num_workers = 1
