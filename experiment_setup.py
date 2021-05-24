import os

ENCODER = "brnn"

model_config = open('model.config.yaml').read()
for folder_name in os.listdir('datasets/'):
    try:
        os.makedirs(os.path.join(ENCODER, folder_name, "prediction"))
        os.makedirs(os.path.join(ENCODER, folder_name, "reports"))
    except OSError:
        pass
    file = open(os.path.join(ENCODER, folder_name, f'{ENCODER}.{folder_name}.yaml'), 'w')
    path_to_save = f"save_data: ../../datasets/{folder_name}/samples\n"
    file.write(path_to_save)
    source_path = f"src_vocab: ../../datasets/{folder_name}/vocab/portuguese.vocab\n"
    file.write(source_path)
    tgt_path = f"tgt_vocab: ../../datasets/{folder_name}/vocab/portuguese.vocab\n"
    file.write(tgt_path)
    options = "overwrite: False\n \
             share_vocab: True\n"
    file.write(options)
    data_str = "data:\n" \
               "corpus_1:\n" \
               f"\tpath_src: ../../datasets/{folder_name}/train.{folder_name.split('-')[0]}\n" \
               f"\tpath_tgt: ../../datasets/{folder_name}/train.{folder_name.split('-')[1]}\n" \
               "valid:\n" \
               f"\tpath_src: ../../datasets/{folder_name}/val.{folder_name.split('-')[0]}\n" \
               f"\tpath_tgt: ../../datasets/{folder_name}/val.{folder_name.split('-')[1]}\n"
    file.write(data_str)
    model_path = f"save_model: {ENCODER}/run/{folder_name}/model\n"
    file.write(model_path)
    encoder = f"encoder_type: {ENCODER}\n"
    file.write(encoder)
    file.write(model_config)
    file.close()

for folder_name in os.listdir(ENCODER):
    config_file = os.path.join(ENCODER, folder_name, ENCODER+'.'+folder_name+'.yaml')

    os.system(f'onmt_build_vocab -config {config_file} -n_sample 10000')
    os.system(f'onmt_train -config {config_file}')
    source = folder_name.split('-')[0]
    target = folder_name.split('-')[1]
    test_file = f"datasets/{folder_name}/test.{source}"
    os.system(f'onmt_translate -model {ENCODER}/run/{folder_name}/model_step_30000.pt -src {test_file}'
              f'-output prediction/{source}-{target}-pred.txt -gpu 0 -verbose')

