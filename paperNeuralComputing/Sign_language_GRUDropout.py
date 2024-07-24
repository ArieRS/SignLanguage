import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from datetime import datetime
#####
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


def GRU_model(X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                pathWorkspace,
                learning_rate_param,
                epochs, 
                batch_size,
                output_class,
                model_name="best_model/best_model_gru.keras"):
  
    start_time = datetime.now()
    model = Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add((GRU(128, return_sequences=True, activation='relu')))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add((GRU(64, return_sequences=True, activation='relu')))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add((GRU(32, return_sequences=False, activation='relu')))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_class, activation='softmax'))

    optimizer = Adam(learning_rate=learning_rate_param, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy',
                           tf.keras.metrics.F1Score(
                                average="weighted", name='f1_score')
                           ])

    '''### early stop and model checkpoint'''
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    mc = ModelCheckpoint(
                         os.path.join(pathWorkspace, model_name), 
                         monitor='val_accuracy', mode='max', verbose=1, 
                         save_best_only=True)


    runWandb = wandb.init(
                    project="paperNeuralComputing",
                    entity="army",
                    config={
                             "learning_rate": learning_rate_param,
                             "architecture": "GRU",
                             "dataset": output_class,
                             "epochs": epochs}
                )

    # get the start time
    history = model.fit(X_train, y_train, 
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), 
                        callbacks=[
                            mc,
                            WandbMetricsLogger()]
                        )
    
    end_time = datetime.now()
    durationTraining = end_time - start_time

    start_time = datetime.now()
    # load the saved model
    saved_model = load_model(os.path.join(pathWorkspace, model_name))
    end_time = datetime.now()
    durationTesting = end_time - start_time
    print(saved_model.summary())
    runWandb.finish()

    # evaluate model  
    loss, accuracy, f1 = saved_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    return loss, accuracy, f1, durationTraining, durationTesting


def summarize_results(scores, 
                      all_loss, 
                      all_f1,
                      training_time_list, 
                      testing_time_list):
    # summarize mean and standard deviation
    max_score = 0
    
    for i in range(len(scores)):
        if scores[i] > max_score:
            max_score = scores[i]

    mean, std = np.mean(scores), np.std(scores)
    meanLoss, stdLoss = np.mean(all_loss), np.std(all_loss)
    meanLF1, stdF1 = np.mean(all_f1), np.std(all_f1)
    meanTrainingTime = np.mean(training_time_list)
    meanTestingTime = np.mean(testing_time_list)

    print(f'Max Score = {max_score} Mean Score = {mean} Std Deviation = {std}')
    print(f'Mean F1 = {meanLF1} Std Deviation F1 = {stdF1}')
    print(f'MSE = {meanLoss} Std Deviation MSE = {stdLoss}')
    print(f'Mean Training Time = {meanTrainingTime}')
    print(f'Mean Testing Time = {meanTestingTime}')

    # boxplot of scores
    # plt.boxplot(scores)
    # plt.savefig('./{}.png'.format(box_plot_save_name))


def run_experiment(X_train, y_train,
                   X_val, y_val, 
                   X_test, y_test, 
                   pathWorkspace,
                   learning_rate_param=0.0001,
                   batch_size=32, epochs=100, repeats=3, output_class=100):
    all_score = list()
    all_loss = list()
    all_f1 = list()
    all_training_time = list()
    all_testing_time = list()

    for _ in range(repeats):
        loss, score, f1, duration_train, duration_test = GRU_model( 
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test, pathWorkspace,
                          learning_rate_param=learning_rate_param,
                          batch_size=batch_size, 
                          epochs=epochs,
                          output_class=output_class,
                          model_name="best_model/best_model_gru.keras")
        score = score * 100
        tempLoss = loss
        all_score.append(score)
        all_loss.append(tempLoss)
        all_f1.append(f1)
        all_training_time.append(duration_train)
        all_testing_time.append(duration_test)

    summarize_results(all_score, 
                      all_loss, 
                      all_f1,
                      all_training_time, 
                      all_testing_time)


if __name__ == '__main__':
    # to set GPU off
    # tf.config.set_visible_devices([], 'GPU')
    sequencesTraining = []
    labelsTraining = []

    # 10 class 78 frames
    pathWorkspace = '/home/bra1n/Documents/signLanguage/paperNeuralComputing'
    variableClass = '100'

    fileNameNpy = 'DataSaveOnNumpy/{}AllFrame_WLASL_{}Class_option2.npy'.format
    fileNameLabel = 'DataSaveOnNumpy/{}LabelAllFrame_WLASL_{}Class_option2.npy'.format
    
    pathKeypointTraining = os.path.join(pathWorkspace, fileNameNpy("Training",variableClass))
    pathLabelTraining = os.path.join(pathWorkspace, fileNameLabel("Training", variableClass))
    pathKeypointVal = os.path.join(pathWorkspace, fileNameNpy("Validation", variableClass))
    pathLabelVal = os.path.join(pathWorkspace, fileNameLabel("Validation", variableClass))
    pathKeypointTest = os.path.join(pathWorkspace, fileNameNpy("Testing", variableClass))
    pathLabelTest = os.path.join(pathWorkspace, fileNameLabel("Testing", variableClass))
    
    X_train = np.load(pathKeypointTraining, allow_pickle=True)
    y_train = np.load(pathLabelTraining, allow_pickle=True)
    X_val = np.load(pathKeypointVal, allow_pickle=True)
    y_val = np.load(pathLabelVal, allow_pickle=True)
    X_test = np.load(pathKeypointTest, allow_pickle=True)
    y_test = np.load(pathLabelTest, allow_pickle=True)

    # convert String into integer encode
    labelEncoder = LabelEncoder()
    y_train_labelEncoder = labelEncoder.fit_transform(y_train)
    y_val_labelEncoder = labelEncoder.fit_transform(y_val)
    y_test_labelEncoder = labelEncoder.fit_transform(y_test)

    # convert integer encode into binary
    oneHotEncoder = OneHotEncoder(sparse_output=False)
    yTrainLabel_integer_encode = y_train_labelEncoder.reshape(len(y_train_labelEncoder),1)
    yTrainLabel_onehot_encode = oneHotEncoder.fit_transform(yTrainLabel_integer_encode)

    yValLabel_integer_encode = y_val_labelEncoder.reshape(len(y_val_labelEncoder),1)
    yValLabel_onehot_encode = oneHotEncoder.fit_transform(yValLabel_integer_encode)

    yTestLabel_integer_encode = y_test_labelEncoder.reshape(len(y_test_labelEncoder),1)
    yTestLabel_onehot_encode = oneHotEncoder.fit_transform(yTestLabel_integer_encode)
    #######

    xp = np
    configParam = {}
    configParam['learning_rate_param'] = 0.0001
    configParam['batch_size'] = 32
    configParam['epochs'] = 150
    configParam['repeats'] = 5
    configParam['output_class'] = 100

    run_experiment(
        X_train, yTrainLabel_onehot_encode, 
        X_val, yValLabel_onehot_encode,
        X_test, yTestLabel_onehot_encode, 
        pathWorkspace, 
        learning_rate_param=configParam['learning_rate_param'], 
        batch_size=configParam['batch_size'], 
        epochs=configParam['epochs'], 
        repeats=configParam['repeats'],
        output_class=configParam['output_class'])
    
    print(f'{X_train.shape=};{yTrainLabel_onehot_encode.shape=};{X_test.shape=};{yTestLabel_onehot_encode.shape}')
