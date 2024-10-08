# Sign Language Recognition

## Steps to Run the Code

1. Set the directory in each code, then run the program in the order as below.
2. `LoadData.py` has function to copy video dataset from WLASL dataset into target folder
3. `ExtractTheKeypoint.py` uses to extract the keypoint from video dataset, write on csv the video name, total extracted frames and the action labels
4. `LoadExtractedKeypoint.py` has function to load the data and save into `.npy`
5. Run the classification algorithm such as `Sign_language_GRUDropout.py`, `Sign_language_BiGRUDropout.py`, `ReservoirRewriteMulti_withoutOptunaEdit.py`, etc

## How to Access the Dataset

Please access the dataset from https://dxli94.github.io/WLASL/
