from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import losses
from sentence_transformers import InputExample
from sentence_transformers.evaluation import TripletEvaluator

import os

### load dataset
class LoadDataset:
    def __init__(self, dataset_train_path, dataset_val_path):
        self.dataset_train_path = dataset_train_path
        self.dataset_val_path = dataset_val_path

    def load_dataset(self, season_num):
        dataset_id_train = self.dataset_train_path + '/' \
                           + 'onlyconnect_triplet_train_season_' + str(season_num) + '.jsonl'
        dataset_id_val = self.dataset_val_path + '/' \
                           + 'onlyconnect_triplet_val_season_' + str(season_num) + '.jsonl'
        # load dataset
        dataset = load_dataset("json", data_files={"train": dataset_id_train, "validation": dataset_id_val})

        train_examplesB = []
        train_dataB = dataset['train']['set']
        n_examples = dataset['train'].num_rows
        for i in range(n_examples):
            example = train_dataB[i]
            train_examplesB.append(InputExample(texts=[example['query'], example['pos'], example['neg']]))

        validation_examplesB = []
        validation_dataB = dataset['validation']['set']
        n_examples_validation = dataset['validation'].num_rows
        for i in range(n_examples_validation):
            example = validation_dataB[i]
            validation_examplesB.append(InputExample(texts=[example['query'], example['pos'], example['neg']]))

        return train_examplesB, validation_examplesB

### pretrained model loader
class PretrainedModelLoader:
    def __init__(self, model_name='bert'):
        self.model_name = model_name
        # self.MODEL_LST = ['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased']
        # self.MODEL_LST_ST = ["all-mpnet-base-v2"]

    def load_model(self):
        # identifier = ['bert', 'roberta', 'distilbert', 'mpnet']
        # if self.model_name not in identifier:
        #     raise ValueError('model name not supported')
        if self.model_name.split('/')[0] == 'sentence-transformers' or self.model_name.split('/')[0] == 'microsoft':
            # load pretrained model from sentence transformers
            model = SentenceTransformer(self.model_name)
        else:
            # if not sentence transformer:
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        return model

### fine-tune transformers with implementation of one-season-out cross validation
class TransformerTrainer:
    def __init__(self, model_save_dir='/media/saboa/DATA/nlp', train_dataset_dir='./dataset',
                 val_dataset_dir='./dataset', model_name='bert', batch_size=16, num_epochs=10, num_seasons=15):
        self.model_name = model_name
        self.model_save_dir = model_save_dir
        self.train_dataset_dir = train_dataset_dir
        self.val_dataset_dir = val_dataset_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_seasons = num_seasons


    def train(self):
        # load dataset
        dataset = LoadDataset(self.train_dataset_dir, self.val_dataset_dir)
        for season in range(1, self.num_seasons+1):
            model_save_path = self.model_save_dir + '/' + self.model_name.split('/')[-1] + '/' + str(season)
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            train_examplesB, validation_examplesB = dataset.load_dataset(season)
            # load train dataset
            train_dataloaderB = DataLoader(train_examplesB, shuffle=True, batch_size=self.batch_size)
            dev_evaluator = TripletEvaluator.from_input_examples(validation_examplesB,
                                                                 batch_size=self.batch_size, name='sts-dev')
            model = PretrainedModelLoader(self.model_name).load_model()
            train_lossB = losses.TripletLoss(model=model, triplet_margin=1)
            warmup_stepsB = int(len(train_dataloaderB) * self.num_epochs * 0.1)  # 10% of train data
            print('current season validation: ', season)
            # callbacks = keras.callbacks.ModelCheckpoint(filepath=model_save_path, monitor='val_loss', verbose=0, mode='auto')
            model.fit(train_objectives=[(train_dataloaderB, train_lossB)],
                      evaluator=dev_evaluator,
                      epochs=self.num_epochs,
                      evaluation_steps=int(len(train_dataloaderB) * 0.1),
                      warmup_steps=warmup_stepsB,
                      output_path=model_save_path,
                      save_best_model=True)
            model_temp = SentenceTransformer(model_save_path)
            triplet_accuracy = model_temp.evaluate(dev_evaluator)
            print("Season " + str(season) + " validation accuracy: " + str(triplet_accuracy))


if __name__ == '__main__':
    trainer = TransformerTrainer(model_name='microsoft/deberta-v3-base')
    trainer.train()