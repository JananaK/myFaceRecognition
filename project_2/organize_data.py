import shutil
import os
import glob

path = '/home/pi/myFaceRecognition/images/'
imgFolders = glob.glob('/home/pi/myFaceRecognition/images/*/')
#imgFolders = listdirs(path)

folder_num = 0
folder_string = '0'

for x in range (0, 19):
    img_count = 0
    
    if folder_num < 10:
        folder_string = str(folder_num)
        folder_string = folder_string.zfill(2)
    else:
        folder_string = str(folder_num)

    eachFolder = path + folder_string

    num_files = len([name for name in os.listdir(eachFolder) if os.path.isfile(os.path.join(eachFolder, name))])
    print(num_files)

    train_count = num_files * .7
    vali_count = num_files * .1
    
    test_count = num_files * .2

    folder_num += 1

    for eachImg in os.listdir(eachFolder):
        img_path = eachFolder + '/' + eachImg
        if img_count < train_count:
            destination = "/home/pi/myFaceRecognition/data/Train/" + folder_string + "/"
            shutil.copy(img_path, destination)
        elif img_count < train_count + vali_count:
            destination = '/home/pi/myFaceRecognition/data/Validation/' + folder_string + '/'
            shutil.copy(img_path, destination)
        else:
            destination = '/home/pi/myFaceRecognition/data/Test/' + folder_string + '/'
            shutil.copy(img_path, destination)

        img_count += 1

