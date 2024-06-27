import numpy as np
import os
import os.path
import tslearn
from tslearn.preprocessing import TimeSeriesResampler

class LoadExtractedKeypoint:

  def LoadKeypoint(self, pathWorkspace, folder_path_keypoint, file_name_data_numpy, file_name_label_numpy):
    
    xTemp = []
    yTemp = []
  
    DATA_PATH_KEYPOINT = os.path.join (pathWorkspace, folder_path_keypoint)
    
    actionsVideoInput = np.array(os.listdir(DATA_PATH_KEYPOINT))
     
    for action in actionsVideoInput:
      print ("action = {}".format(action))
      for actionSubFolder in np.array(os.listdir(os.path.join(DATA_PATH_KEYPOINT, action))):
        sequences = 0
        window = []
        tempSequencesWhichExist = 0
        # list all npy file for each frame in the video
        allKeyPointData = os.listdir(os.path.join(DATA_PATH_KEYPOINT, action, actionSubFolder))
        for _ in range (len(allKeyPointData)): 
            pathFile = os.path.join(DATA_PATH_KEYPOINT, action, actionSubFolder, "{}.npy".format(sequences))
          
            # if file not exist take the previous keypoint 
            if (os.path.isfile(pathFile)==False):
              pathFile = os.path.join(DATA_PATH_KEYPOINT, action, actionSubFolder, "{}.npy".format(tempSequencesWhichExist))
            else:
              tempSequencesWhichExist = sequences
            
            res = np.load(pathFile)
            window.append(res)
            sequences +=1
        
        ### interpolation in here
        window        = np.asarray(window, dtype=object)
        array_reshape = np.reshape(window, window.shape[0]* window.shape[1])
        size          = 78 * window.shape[1]
        result_interpolation  = TimeSeriesResampler(sz = size).fit_transform(array_reshape) 
        final_window  = np.reshape(result_interpolation, (78, window.shape[1]))

        xTemp.append(final_window)
        yTemp.append(action)
    
    ### save keypoint
    pathKeypoint = os.path.join(pathWorkspace,'DataSaveOnNumpy',file_name_data_numpy)
    np.save (pathKeypoint, xTemp)
    pathLabel = os.path.join(pathWorkspace,'DataSaveOnNumpy',file_name_label_numpy)
    np.save (pathLabel, yTemp) 
  
if __name__ == "__main__":
  BASE_Dir = "/home/bra1n/Documents/signLanguage"
  PATH_WORKSPACE = '/home/tamlab/Documents/SignLanguage/CodeInServer/Project2'
 
  # FOLDER_PATH_TRAINING  = 'KeypointTrainingWLASL100'
  # FOLDER_PATH_TESTING   = 'KeypointTestingWLASL100'

  FOLDER_PATH_TRAINING = 'KeypointTrainingWLASL100_option2'
  FOLDER_PATH_TESTING = 'KeypointTestingWLASL100_option2'

  FILE_NAME_TRAINING_NUMPY = 'TrainingAllFrame_WLASL_100Class_option2'
  FILE_NAME_LABEL_TRAINING_NUMPY = 'TrainingLabelAllFrame_WLASL_100Class_option2'

  FILE_NAME_TEST_NUMPY = 'TestingAllFrame_WLASL_100Class_option2'
  FILE_NAME_LABEL_TEST_NUMPY = 'TestingLabelAllFrame_WLASL_100Class_option2'

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
                              FOLDER_PATH_TESTING, 
                              FILE_NAME_TEST_NUMPY, 
                              FILE_NAME_LABEL_TEST_NUMPY)

  # ### list training folder
  # actionsTraining = np.array(os.listdir(DATA_PATH_KEYPOINT_TRAINING))
  # print("Load data in process ")

  # ### hard code the array length because the array has different step
  # ### xTraining =  np.empty(1780, dtype=object)
  # ### yTraining =  np.empty(1780, dtype=object)
  # xTraining = []
  # yTraining = []
  # indexVideoTraining = 0


  # for action in actionsTraining:
  #   print ("action = {}".format(action))
  #   for actionSubFolder in np.array(os.listdir(os.path.join(DATA_PATH_KEYPOINT_TRAINING, action))):
  #     sequences = 0
  #     window = []
  #     tempSequencesWhichExist = 0
  #     print (actionSubFolder)
  #     ### list all npy file for each frame in the video
  #     allKeyPointData = os.listdir(os.path.join(DATA_PATH_KEYPOINT_TRAINING, action, actionSubFolder))
  #     for _ in range (len(allKeyPointData)): 
  #         pathFile = os.path.join(DATA_PATH_KEYPOINT_TRAINING, action, actionSubFolder, "{}.npy".format(sequences))
        
  #         # if file not exist take the previous keypoint 
  #         if (os.path.isfile(pathFile)==False):
  #           pathFile = os.path.join(DATA_PATH_KEYPOINT_TRAINING, action, actionSubFolder, "{}.npy".format(tempSequencesWhichExist))
  #         else:
  #           tempSequencesWhichExist = sequences
          
  #         res = np.load(pathFile)
  #         window.append(res)
  #         sequences +=1
  #         # print("window  = ",np.array(window).shape)
  #     ### sequencesTraining.append(window)
  #     ### labelsTraining.append(action)
  #     ### xTraining[indexVideoTraining] = window
  #     ### yTraining[indexVideoTraining] = action

  #     ### interpolation in here
  #     window        = np.asarray(window, dtype=object)
  #     array_reshape = np.reshape(window, window.shape[0]* window.shape[1])
  #     size          = 78 * window.shape[1]
  #     result_interpolation  = TimeSeriesResampler(sz = size).fit_transform(array_reshape) 
  #     final_window  = np.reshape(result_interpolation, (78, window.shape[1]))
    
  #     xTraining.append(final_window)
  #     yTraining.append(action)

  #     indexVideoTraining += 1
  #     ### print("Shape each frame = ",np.array(sequencesTraining).shape)
  #     ### print("Label each video = ",np.array(labelsTraining).shape)
    
  # ## save keypoint
  # pathKeypoint  = os.path.join(pathWorkspace, 'DataSaveOnNumpy',  PATH_SAVE_TRAINING_NUMPY)
  # np.save (pathKeypoint, xTraining)
  # pathLabel     = os.path.join(pathWorkspace, 'DataSaveOnNumpy',  PATH_SAVE_LABEL_TRAINING_NUMPY)
  # np.save (pathLabel, yTraining) 

  # ###data Testing
  # print("Load data in process ")
  # # list testing folder
  # actionsTesting  = np.array(os.listdir(DATA_PATH_KEYPOINT_TESTING))

  # ### xTesting = np.empty(258, dtype=object)
  # ### yTesting = np.empty(258, dtype=object)
  # xTesting = []
  # yTesting = []
  # indexVideoTesting = 0
  
  # for action in actionsTesting:
  #   for actionSubFolder in np.array(os.listdir(os.path.join(DATA_PATH_KEYPOINT_TESTING, action))):
  #     sequences = 0
  #     window = []
  #     tempSequencesWhichExist = 0
  #     allKeyPointData = os.listdir(os.path.join(DATA_PATH_KEYPOINT_TESTING, action, actionSubFolder))
  #     for _ in range (len(allKeyPointData)): 
  #         ## print (os.path.join(DATA_PATH_KEYPOINT_TESTING, action, actionSubFolder, "{}.npy".format(sequence) ) )
  #         pathFile = os.path.join(DATA_PATH_KEYPOINT_TESTING, action, actionSubFolder, "{}.npy".format(sequences))
  #         if (os.path.isfile(pathFile)==False):
  #           pathFile = os.path.join(DATA_PATH_KEYPOINT_TESTING, action, actionSubFolder, "{}.npy".format(tempSequencesWhichExist))
  #         else:
  #           tempSequencesWhichExist = sequences
          
  #         res = np.load(pathFile)
  #         window.append(res)
  #         sequences +=1
  #     ### this for frame which have different frame
  #     ### sequencesTesting.append(window)
  #     ### labelsTesting.append(action)
  #     ### xTesting[indexVideoTesting] = window
  #     ### yTesting[indexVideoTesting] = action

  #     ### interpolation in here
  #     window        = np.asarray(window)
  #     array_reshape = np.reshape(window, window.shape[0]* window.shape[1])
  #     size          = 78 * window.shape[1]
  #     result_interpolation  = TimeSeriesResampler(sz = size).fit_transform(array_reshape) 
  #     final_window  = np.reshape(result_interpolation, (78, window.shape[1]))


  #     xTesting.append(final_window)
  #     yTesting.append(action)

  #     indexVideoTesting += 1

  # ## save keypoint
  # pathKeypoint  = os.path.join(pathWorkspace, 'DataSaveOnNumpy',  PATH_SAVE_TEST_NUMPY)
  # np.save (pathKeypoint, xTesting)
  # pathLabel     = os.path.join(pathWorkspace, 'DataSaveOnNumpy',  PATH_SAVE_LABEL_TEST_NUMPY)
  # np.save (pathLabel, yTesting)

