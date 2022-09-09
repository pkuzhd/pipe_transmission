import numpy as np
import cv2
import os
import json

path = "./DTU/scan1/"
savePath = "./DTU/save/"
imgIndex = [31,32,33,34,35]


def para():
    camPath = "cams_1/"
    dict = {}
    
    for i,index in enumerate(imgIndex):
        dict[str(i + 1)] = {}
        dict[str(i + 1)]["K"] = None
        dict[str(i + 1)]["R"] = None
        with open(path+camPath+'%08d'%index+"_cam.txt","r") as f:
            lines = f.readlines()
            for line in lines:
                if line == "extrinsic\n":
                    offset = "R"
                    matrix = []
                elif offset == "R":
                        tmp = line.split()
                        for j in range(len(tmp)):
                            tmp[j] = float(tmp[j])
                        matrix.append(tmp)
                        if len(matrix) == 4:
                            for k in range(3):
                                matrix[k][3] /= 1000
                            matrix = np.asarray(matrix)
                            if i == 0:  R0 = matrix
                            matrix =  matrix @ np.linalg.inv(R0) 
                            dict[str(i + 1)][offset] = matrix.tolist()
                            offset = None
                elif line == "intrinsic\n":
                    offset = "K"
                    matrix = [] 
                elif offset == "K":
                    tmp = line.split()
                    for j in range(len(tmp)):
                        tmp[j] = float(tmp[j])
                    matrix.append(tmp)
                    if len(matrix) == 3:
                        dict[str(i + 1)][offset] = matrix
                        offset = None
        f.close()
    saveJson(dict)
    return dict

def saveJson(dict):
    try:
        os.mknod(savePath+"paraDTU.json")
    except FileExistsError:
        os.remove(savePath+"paraDTU.json")
        os.mknod(savePath+"paraDTU.json")
    with open(savePath+"paraDTU.json","w") as f:
    #os.write(f,jsonData)
        jsonData = json.dump(dict,fp=f)
    f.close()
    print("finish write !")
                    
if __name__  == "__main__":
    oldcenter = [
            [0.,         0.,          0. , 1],
            [0.13442005, -0.00617611, 0.00693867 , 1],
            [0.26451552, -0.01547954, 0.01613046 , 1],
            [0.39635817, -0.01958267, 0.02322004 , 1],
            [0.53632535, -0.02177305, 0.05059797 , 1]
    ]
    dict = para()
    with open("../data/para.json","r") as fp:
        oridict = json.load(fp)
    newCenter = []
    for i in range(len(dict)):
        Rnew = np.asarray(dict[str(i+1)]["R"])
        #Rold = np.asarray(oridict[str(i+1)]["R"])
        #Rnewer = np.linalg.inv(Rnew) @ Rold
        #newCenter.append(Rnewer@np.asarray(oldcenter[i]).T)
        newCenter.append(Rnew@np.array([0,0,0,1]).T)
    
    #dict[str(i+1)]["R"] = Rnewer.tolist()
    #saveJson(dict)
    print("{")
    for i in range(len(newCenter)):
        if i < 4:
            print("{",newCenter[i][0],",",newCenter[i][1],",",newCenter[i][2],"}",",")
        else:
            print("{",newCenter[i][0],",",newCenter[i][1],",",newCenter[i][2],"}")
    print("}")

    print("start mask")
    imgPath = "mask/"
    for i,index in enumerate(imgIndex):
        img = cv2.imread(path+imgPath+'%08d'%index+"_final.png",cv2.IMREAD_UNCHANGED)
        bytesImg = img.tobytes()   
        mask = np.ones([1200,1600],dtype=np.uint8) * 255
        cv2.imwrite(savePath+imgPath+"1-" + str(i + 1) +".png",img)
        print(i,"is finish")
    
    print("start depth")
    imgPath = "depth/"
    depPath = "depth_est/"
    for i,index in enumerate(imgIndex):
        img = cv2.imread(path+depPath+'%08d'%index+".pfm",cv2.IMREAD_UNCHANGED) / 1000
        bytesImg = img.tobytes()   
        os.mknod(savePath+imgPath+"1-" + str(i + 1) +".depth")
        f = os.open(savePath+imgPath+"1-" + str(i+ 1) +".depth",os.O_WRONLY)
        os.write(f,bytesImg)
        os.close(f)
        #cv2.imwrite(savePath+imgPath+"1-" + str(i + 1) +".depth",img)
        print(i,"is finish")
        
    print("start imgs")
    pathname = "images/"
    imgPath = "video/"
    for i,index in enumerate(imgIndex):
        img = cv2.imread(path+pathname+'%08d'%index+".jpg",cv2.IMREAD_UNCHANGED)
        bytesImg = img.tobytes()   
        cv2.imwrite(savePath+imgPath+"1-" + str(i+ 1) +".png",img)
        print(i,"is finish")
    