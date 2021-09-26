import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
import math
from ast import literal_eval
import copy
from scipy.optimize import least_squares
import time

from transformations import Euler_ZXZ_Matrix, minimize_Reprojection, generate_3D_Points


# MIN_MATCH_COUNT = 10

# curr_R = None
# curr_t = None

# ROUNDABOUT

# LOW Q
# Execution times:
# SIFT: 1919.83975697s
# SURF: 1811.03619003s
# FAST: 1770.35452604s
# ORB: 1865.84989715s
FIRST_FRAME = 4077
NO_OF_FRAMES = 251
WCorr = -3.032991886138916
HCorr = -59.0875129699707

# HIGH Q
# Execution times:
# SIFT: 3653.45950294s
# SURF: 3964.37969589s
# FAST: 2789.57575798s
# ORB: 1765.94023609s
# FIRST_FRAME = 9690
# NO_OF_FRAMES = 250
# WCorr = -2.9709365367889404
# HCorr = -58.88513946533203


# STRAIGHT 
# WCorr = -6.949899673461914
# HCorr = 85.45460510253906
# HIGH Q

# SUNNY
# Execution times:
# SIFT: 5212.5736742s
# SURF: 4488.48341703s
# FAST: 3546.10450506s
# ORB: 1549.608114s
# FIRST_FRAME = 1809
# NO_OF_FRAMES = 230

# DYNAMIC1
# Execution times:
# SIFT: 4713.30974984s
# SURF: 3761.49601102s
# FAST: 3398.83523417s
# ORB: 772.528777122s zastao na 1413
# FIRST_FRAME = 1295
# NO_OF_FRAMES = 231


translation = None
rotation = None
calc_cords = []

ROUTE = 'roundabout_route'
QUALITY = 'low'
WEATHER = 'sunny'

alg_name = 'orb'

def sift(img, save_image=False):
    # Applying SIFT detector
    sift = cv.xfeatures2d.SIFT_create() 
    kp, des = sift.detectAndCompute(img, None)
    
    if save_image:
        # Marking the keypoint on the image using circles
        img=cv.drawKeypoints(img ,
                            kp ,
                            img ,
                            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        dirname = '_sift'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        cv.imwrite(os.path.join(dirname, "{}.jpg".format(i)), img)
    
    return kp, des


def surf(img):
    surf = cv.xfeatures2d.SURF_create(400) 
    kp, des = surf.detectAndCompute(img, None)
    return kp, des


def orb(img):
    orb = cv.ORB_create()

    # set parameters 
    # orb.setScoreType(cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
    # orb.setWTA_K(3)

    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def fast(img):
    fast = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kp = fast.detect(img)
    # using brisk for descriptors since fast only detects keypoints
    br = cv.BRISK_create()
    kp, des = br.compute(img,  kp)  # note: no mask here!
    return kp, des


def brute_force_matcher(des1, des2, alg='sift'):

    # create BFMatcher object
    if (alg == 'surf'):
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    elif (alg == 'sift'):
        bf = cv.BFMatcher()
    elif (alg == 'orb'):
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    elif (alg == 'fast'):
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return matches


def LK_feature_tracking(prev_img, curr_img, prev_kp):

    prev_kp = np.array([x.pt for x in prev_kp], dtype=np.float32)

    lk_params = dict( winSize  = (15,15),
                          maxLevel = 3,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.03))

    curr_kp, st, err = cv.calcOpticalFlowPyrLK(prev_img, curr_img, prev_kp, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = prev_kp[st == 1]
    kp2 = curr_kp[st == 1]

    return kp1, kp2, err


def extract_features(img, alg='sift'):
    # not good
    if (alg == 'orb'):
       kp, des = orb(img) 
    elif (alg == 'sift'):
       kp, des = sift(img) 
    elif (alg == 'surf'):
        kp, des = surf(img)
    elif (alg == 'fast'):
        kp, des = fast(img)
    return kp, des


def compute_location(prev_img, curr_img, prev_kp, curr_kp, prev_des, curr_des, alg='sift', matcher='LK'):   

    if (matcher == 'BF'):
        # extract points
        prev_pts = []
        curr_pts = []
        # compute keypoint matches using descriptor
        matches = brute_force_matcher(prev_des, curr_des, alg=alg)
       
        for i,(m) in enumerate(matches):
            if m.distance < 20:
                #print(m.distance)
                curr_pts.append(curr_kp[m.trainIdx].pt)
                prev_pts.append(prev_kp[m.queryIdx].pt)
        prev_pts  = np.asarray(prev_pts)
        curr_pts = np.asarray(curr_pts)
    elif (matcher == 'LK'):
        prev_pts, curr_pts, _ = LK_feature_tracking(prev_img, curr_img, prev_kp)

    
    # Compute fundamental/ essential matrix
    # F, mask = cv.findFundamentalMat(curr_pts, prev_pts, cv.FM_8POINT)
    E, mask = cv.findEssentialMat(curr_pts, prev_pts, focal=f, pp=pp, method=cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv.recoverPose(E, curr_pts, prev_pts, focal=f, pp = pp)
    return R, t


def disparity(left_img, right_img, plt_img=False):
    block = 11
    P1 = block * block * 8
    P2 = block * block * 32
    disparityEngine = cv.StereoSGBM_create(minDisparity=0,numDisparities=32, blockSize=block, P1=P1, P2=P2)
    disparity = disparityEngine.compute(left_img, right_img).astype(np.float32)
    if(plt_img):
        print(disparity)
        plt.imshow(disparity, 'gray')
        plt.show()
    return disparity


def triangulate(left_img, right_img, prev_left_img, prev_right_img, alg='sift'):
    K = np.array([[f, 0, pp[0]],
                  [0, f, pp[1]],
                  [0, 0, 1]])

    kp_left, des_left = extract_features(left_img, alg)
    kp_prev_left, des_prev_left = extract_features(prev_left_img, alg)

    matches = brute_force_matcher(des_left, des_prev_left)


    # detecting SIFT features in each image and then matching them through their descriptors
    kp_left, des_left = extract_features(left_img, alg)
    kp_right, des_right = extract_features(right_img, alg)

    matches = brute_force_matcher(des_left, des_right)


    R, t = compute_location(left_img, right_img, kp_left, kp_right, des_left, des_right, alg, matcher='LK')
    
    M1 = np.c_[np.eye(3), np.zeros((3, 1))]
    M2 = np.c_[R, t]


    pt_left = np.array([kp_left[matches[i].queryIdx].pt for i in range(len(matches))])
    pt_right = np.array([kp_right[matches[i].trainIdx].pt for i in range(len(matches))])

    print(pt_left)
    print(K)

    points1u = cv.undistortPoints(pt_left.T, cameraMatrix=K, distCoeffs=None)
    points2u = cv.undistortPoints(pt_right.T, cameraMatrix=K, R=R, distCoeffs=None)

    points3d = cv.triangulatePoints(np.matmul(K, M1), np.matmul(K, M2), points1u, points2u)

    points3d /= points3d[3]
    np.savetxt('data.csv', points3d.T, delimiter=',')
    return points3d[:3].T, pt_left, pt_right

try: 
    start = time.time()
    for k in range(FIRST_FRAME, FIRST_FRAME + NO_OF_FRAMES):
        
        print(k)
        
        # img = cv.imread('_out/%d.png' %i)
        img = cv.imread('%s/%s/_out_%s_q/%d.png' %(ROUTE, WEATHER, QUALITY, k), 0)

        img_left = cv.imread('%s/%s/_out_left_%s_q/%d.png' %(ROUTE, WEATHER, QUALITY, k), 0)
        img_right = cv.imread('%s/%s/_out_right_%s_q/%d.png' %(ROUTE, WEATHER, QUALITY, k), 0)

        if (img is None):
            continue

        if k == FIRST_FRAME:
            IM_HEIGHT, IM_WIDTH = img.shape
            f = IM_WIDTH / (2 * math.tan(90 * (math.pi / 360)))
            pp = (IM_WIDTH/2, IM_HEIGHT/2)

            Proj1 = np.array([[f, 0.0, pp[0], -0.5*f],[0.0, f, pp[1], 0.0],[0, 0, 1, 0]])
            Proj2 = copy.deepcopy(Proj1)
            Proj1[0][3] = 0
        else:
            prev_image = curr_image
            prev_img_left = curr_img_left
            prev_img_right = curr_img_right
            prev_kp = curr_kp
            prev_des = curr_des
            prev_disparity = curr_disparity

        # in every iteration
        curr_image = img
        curr_img_left = img_left
        curr_img_right = img_right
        
        curr_kp, curr_des = extract_features(curr_image, alg=alg_name)
        curr_disparity = disparity(curr_img_left, curr_img_right)

        if k != FIRST_FRAME:

            ImT1_disparityA = np.divide(prev_disparity, 16.0)
            ImT2_disparityA = np.divide(curr_disparity, 16.0)

            trackPoints1 = cv.KeyPoint_convert(curr_kp)
            trackPoints1 = np.expand_dims(trackPoints1, axis=1)

            lk_params = dict( winSize  = (15,15),
                            maxLevel = 3,
                            criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.03))

            trackPoints2, st, err = cv.calcOpticalFlowPyrLK(prev_image, curr_image, trackPoints1, None, flags=cv.MOTION_AFFINE, **lk_params)

            ptTrackable = np.where(st == 1, 1,0).astype(bool)
            trackPoints1_KLT = trackPoints1[ptTrackable, ...]
            trackPoints2_KLT_t = trackPoints2[ptTrackable, ...]
            trackPoints2_KLT = np.around(trackPoints2_KLT_t)

            error = 4
            errTrackablePoints = err[ptTrackable, ...]
            errThresholdedPoints = np.where(errTrackablePoints < error, 1, 0).astype(bool)
            trackPoints1_KLT = trackPoints1_KLT[errThresholdedPoints, ...]
            trackPoints2_KLT = trackPoints2_KLT[errThresholdedPoints, ...]

    
            hPts = np.where(trackPoints2_KLT[:,1] >= IM_HEIGHT)
            wPts = np.where(trackPoints2_KLT[:,0] >= IM_WIDTH)
            outTrackPts = hPts[0].tolist() + wPts[0].tolist()
            outDeletePts = list(set(outTrackPts))

            if len(outDeletePts) > 0:
                trackPoints1_KLT_L = np.delete(trackPoints1_KLT, outDeletePts, axis=0)
                trackPoints2_KLT_L = np.delete(trackPoints2_KLT, outDeletePts, axis=0)
            else:
                trackPoints1_KLT_L = trackPoints1_KLT
                trackPoints2_KLT_L = trackPoints2_KLT

        
            pointDiff = trackPoints1_KLT_L - trackPoints2_KLT_L
            pointDiffSum = np.sum(np.linalg.norm(pointDiff))

            trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
            trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
            selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])

            disparityMinThres = 0.0
            disparityMaxThres = 100.0

            for i in range(trackPoints1_KLT_L.shape[0]):
                T1Disparity = ImT1_disparityA[int(trackPoints1_KLT_L[i,1]), int(trackPoints1_KLT_L[i,0])]
                T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]

                if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres
                    and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
                    trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
                    trackPoints2_KLT_R[i, 0] = trackPoints2_KLT_L[i, 0] - T2Disparity
                    selectedPointMap[i] = 1

            selectedPointMap = selectedPointMap.astype(bool)
            trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
            trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]
            trackPoints2_KLT_L_3d = trackPoints2_KLT_L[selectedPointMap, ...]
            trackPoints2_KLT_R_3d = trackPoints2_KLT_R[selectedPointMap, ...]

            # 3d point cloud triangulation
            numPoints = trackPoints1_KLT_L_3d.shape[0]
            d3dPointsT1 = generate_3D_Points(trackPoints1_KLT_L_3d, trackPoints1_KLT_R_3d, Proj1, Proj2)
            d3dPointsT2 = generate_3D_Points(trackPoints2_KLT_L_3d, trackPoints2_KLT_R_3d, Proj1, Proj2)
            
            ransacError = float('inf')
            dOut = None
            # RANSAC
            ransacSize = 6
            for ransacItr in range(250):
                sampledPoints = np.random.randint(0, numPoints, ransacSize)
                rD2dPoints1_L = trackPoints1_KLT_L_3d[sampledPoints]
                rD2dPoints2_L = trackPoints2_KLT_L_3d[sampledPoints]
                rD3dPointsT1 = d3dPointsT1[sampledPoints]
                rD3dPointsT2 = d3dPointsT2[sampledPoints]

                dSeed = np.zeros(6)
    
                optRes = least_squares(minimize_Reprojection, dSeed, method='lm', max_nfev=200,
                                    args=(rD2dPoints1_L, rD2dPoints2_L, rD3dPointsT1, rD3dPointsT2, Proj1))

    
                error = minimize_Reprojection(optRes.x, trackPoints1_KLT_L_3d, trackPoints2_KLT_L_3d,
                                                d3dPointsT1, d3dPointsT2, Proj1)

                eCoords = error.reshape((d3dPointsT1.shape[0]*2, 3))
                totalError = np.sum(np.linalg.norm(eCoords, axis=1))

                if (totalError < ransacError):
                    ransacError = totalError
                    dOut = optRes.x


            Rmat = Euler_ZXZ_Matrix(dOut[0], dOut[1], dOut[2])
            translationArray = np.array([[dOut[3]], [dOut[4]], [dOut[5]]])
            

                    
            if (isinstance(translation, np.ndarray)):
                translation = translation + np.matmul(rotation, translationArray)
            else:
                translation = translationArray

            if (isinstance(rotation, np.ndarray)):
                rotation = np.matmul(Rmat, rotation)
            else:
                rotation = Rmat


            x, y = int(-translation[0])+WCorr, int(translation[2])+HCorr
                        
            calc_cords.append([x,y])


        # if i != FIRST_FRAME:
        #     R, t = compute_location(prev_image, curr_image, prev_kp, curr_kp, prev_des, curr_des, alg=alg_name, matcher=matcher)
        #     if (curr_R is None and curr_t is None):
        #         curr_t = t
        #         curr_R = R
        #     else:
        #         curr_t = curr_t + curr_R.dot(t)
        #         curr_R = R.dot(curr_R)  
            
        #     calc_cords.append([curr_t[0][0], curr_t[2][0]])
finally:
    end = time.time()
    print("Execution time: {}s".format(end - start))   
    print("save coordinates to file...")       
    outF = open("%s/%s/calculated_coordinates_%s_q_%s.txt" %(ROUTE, WEATHER, QUALITY, alg_name), "w")
    outF.write(str(calc_cords))
    outF.close()

        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        # flann = cv.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(des_prev,des_curr,k=2)
        # # store all the good matches as per Lowe's ratio test.
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
        # if len(good)>MIN_MATCH_COUNT:
        #     src_pts = np.float32([ kp_prev[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #     dst_pts = np.float32([ kp_curr[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        #     matchesMask = mask.ravel().tolist()
        #     h,w,d = prev_image.shape
        #     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #     dst = cv.perspectiveTransform(pts,M)
        #     curr_image = cv.polylines(curr_image,[np.int32(dst)],True,255,3, cv.LINE_AA)
        # else:
        #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        #     matchesMask = None
        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #            singlePointColor = None,
        #            matchesMask = matchesMask, # draw only inliers
        #            flags = 2)
        # img3 = cv.drawMatches(prev_image,kp_prev,curr_image,kp_curr,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()