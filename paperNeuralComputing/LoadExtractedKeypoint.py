import numpy as np
import os
import os.path
# from tslearn.preprocessing import TimeSeriesResampler
# import tensorflow as tf


class LoadExtractedKeypoint:
    maxFrame = 0

    def setMaxFrame(self, lenghtFrame=0):
        if (lenghtFrame > self.maxFrame):
            self.maxFrame = lenghtFrame

    def getMaxFrame(self):
        return self.maxFrame
   
    def LoadKeypoint(self, pathWorkspace, folder_path_keypoint,
                     file_name_data_numpy, file_name_label_numpy):
        xTemp = []
        newXTemp = []
        yTemp = []
        DATA_PATH_KEYPOINT = os.path.join(pathWorkspace, folder_path_keypoint)
        actionsVideoInput = np.array(os.listdir(DATA_PATH_KEYPOINT))

        for action in actionsVideoInput:
            print("action = {}".format(action))
            for actionSubFolder in np.array(os.listdir(os.path.join(DATA_PATH_KEYPOINT, action))):
                sequences = 1
                window = []
                tempSequencesWhichExist = 0
                # list all npy file for each frame in the video
                allKeyPointData = os.listdir(os.path.join(DATA_PATH_KEYPOINT,
                                                          action,
                                                          actionSubFolder))
                print(f'{allKeyPointData =}')
                for _ in range(len(allKeyPointData)): 
                    print(f'{_}')
                    pathFile = os.path.join(DATA_PATH_KEYPOINT, action,
                                            actionSubFolder,
                                            "{}.npy".format(sequences))
                    
                    print(f'{pathFile=}')
                  
                    # if file not exist take the previous keypoint 
                    if (os.path.isfile(pathFile) is False):
                        pathFile = os.path.join(DATA_PATH_KEYPOINT, action,
                                                actionSubFolder, 
                                                "{}.npy".format(tempSequencesWhichExist))
                    else:
                        tempSequencesWhichExist = sequences
                    
                    res = np.load(pathFile)
                    window.append(res)
                    sequences += 1
                
                # ### interpolation in here
                # window        = np.asarray(window, dtype=object)
                # array_reshape = np.reshape(window, window.shape[0]* window.shape[1])
                # size          = 78 * window.shape[1]
                # result_interpolation  = TimeSeriesResampler(sz = size).fit_transform(array_reshape) 
                # final_window  = np.reshape(result_interpolation, (78, window.shape[1]))
                # window = np.array(window,dtype=object)
                # final_window  = np.transpose(window,[1,0])
                final_window = np.asarray(window)
                # print (f'{np.asarray(final_window).shape=}')

                self.setMaxFrame(lenghtFrame=final_window.shape[0])
                xTemp.append(final_window)
                yTemp.append(action)
        
        # process padding
        print(f'{self.getMaxFrame()=}')
        print(len(xTemp))
        # xTemp = np.asarray(xTemp)
        for jj in range(len(xTemp)):
            tempTranspose = np.transpose(xTemp[jj], [1, 0])
            lenghtPadding = self.getMaxFrame() - tempTranspose.shape[1]
            paddResult = np.pad(tempTranspose, ((0, 0), (0, lenghtPadding)), constant_values=0)
            paddResult = np.transpose(paddResult, [1, 0])
            newXTemp.append(paddResult)

        print(f'{self.getMaxFrame()=}') 
        # save keypoint
        pathKeypoint = os.path.join(pathWorkspace, 'DataSaveOnNumpy', 
                                    file_name_data_numpy)
        # np.save (pathKeypoint, xTemp)
        np.save(pathKeypoint, newXTemp)
        print(np.asarray(newXTemp).shape)
        pathLabel = os.path.join(pathWorkspace, 'DataSaveOnNumpy', 
                                 file_name_label_numpy)
        np.save(pathLabel, yTemp) 


if __name__ == "__main__":
    PATH_WORKSPACE = '/home/bra1n/Documents/signLanguage/paperNeuralComputing'

    FOLDER_PATH_TRAINING = "Keypoint{}WLASL100_normalization_option2".format("Training")
    FOLDER_PATH_VALIDATION = "Keypoint{}WLASL100_normalization_option2".format("Validation")
    FOLDER_PATH_TESTING = "Keypoint{}WLASL100_normalization_option2".format("Testing")

    FILE_NAME_TRAINING_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("Training")
    FILE_NAME_LABEL_TRAINING_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("TrainingLabel")

    FILE_NAME_VALIDATION_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("Validation")
    FILE_NAME_LABEL_VALIDATION_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("ValidationLabel") 

    FILE_NAME_TEST_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("Testing")
    FILE_NAME_LABEL_TEST_NUMPY = "{}AllFrame_WLASL_100Class_option2".format("TestingLabel")

    FOLDER_SAVE_NUMPY = os.path.join(PATH_WORKSPACE, 'DataSaveOnNumpy')
    if not os.path.exists(FOLDER_SAVE_NUMPY):
        os.makedirs(FOLDER_SAVE_NUMPY)

    sequencesTraining = []
    labelsTraining = []
    sequencesTesting = []
    labelsTesting = []

    tempLoadExtractedKeypoint = LoadExtractedKeypoint()

    tempLoadExtractedKeypoint.LoadKeypoint(
                                PATH_WORKSPACE,
                                FOLDER_PATH_TRAINING,
                                FILE_NAME_TRAINING_NUMPY,
                                FILE_NAME_LABEL_TRAINING_NUMPY)
    
    tempLoadExtractedKeypoint.LoadKeypoint(
                                PATH_WORKSPACE,
                                FOLDER_PATH_VALIDATION,
                                FILE_NAME_VALIDATION_NUMPY,
                                FILE_NAME_LABEL_VALIDATION_NUMPY)
    
    tempLoadExtractedKeypoint.LoadKeypoint(
                                PATH_WORKSPACE,
                                FOLDER_PATH_TESTING,
                                FILE_NAME_TEST_NUMPY,
                                FILE_NAME_LABEL_TEST_NUMPY)