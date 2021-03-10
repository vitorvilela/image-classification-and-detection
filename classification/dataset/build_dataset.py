import dataset as ds

ds.build_dataset('./data/images/numbers/train', 'jpg', './npz/service_numbers_train.npz')
ds.plot_dataset('./data/service_numbers_train.npz')

ds.build_dataset('./data/images/numbers/test', 'jpg', './npz/service_numbers_test.npz')
ds.plot_dataset('./data/service_numbers_test.npz')

ds.build_dataset('./data/images/letters/train', 'jpg', './npz/service_letters_train.npz')
ds.plot_dataset('./data/service_letters_train.npz')

ds.build_dataset('./data/images/letters/test', 'jpg', './npz/service_letters_test.npz')
ds.plot_dataset('./data/service_letters_test.npz')
