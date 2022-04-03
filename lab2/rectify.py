import numpy as np
import cv2

# ART: factorize a PPM as P=A*[R;t]
def art(P):
    Q = np.linalg.inv(P[0:3, 0:3])
    U, B = np.linalg.qr(Q)
    R = np.linalg.inv(U)
    t = B @ P[0:3, 3]
    A = np.linalg.inv(B)
    # print("A:", A)
    A = A / A[2,2]
    # print("A_after:", A)
    # print("A, R, t:", A, '\n', R, '\n', t)
    return A, R, t

def getPo():
    A1 = np.array([[934.7, 0.0,       375.18],
                  [0.0,    903.91,   290.0],
                  [0.0,    0.0,       1.0]])
    A2 = np.array([[933.5, 0.0,     377.68],
                   [0.0,    907.12, 287.7],
                   [0.0,    0.0,     1.0]])
    R1 = np.array([[0.8135, -0.0758, 0.5767],
                  [0.0117, 0.9934,  0.1141],
                  [-0.5815,-0.0861, 0.8090]])
    # A1 = A1.T
    # R1 = R1.T
    # print("R1:", R1)
    R2 = np.array([[0.811845,  -0.075058, 0.579028],
                   [0.012814,  0.993754,  0.110851],
                   [-0.583732, -0.082574, 0.807736]])
    t1 = np.array([[-336.054], [-31.3943], [-1207.7]])
    t2 = np.array([[-623.831787], [-37.058510], [-932.469971]])

    # "A[0][2] = A[0][2] + 160" aims to keep the rectified image in the
    # center of the 768 × 576 window
    A1[0][2] = A1[0][2] + 160
    A2[0][2] = A2[0][2] + 160
    print(A1)
    Po1 = A1 @ np.append(-R1, t1, axis=1)
    Po2 = A2 @ np.append(-R2, t2, axis=1)

    print("Po1:\n", Po1)
    print("Po2:\n", Po2)
    return Po1, Po2

# def rectify(Po1, Po2, A1, A2, R1):
def rectify(Po1, Po2):
    # factorize old PPMs
    A1, R1, t1 = art(Po1)
    A2, R2, t2 = art(Po2)

    # optical centers (unchanged)
    c1 = -np.linalg.inv(Po1[:, 0:3]) @ Po1[:, 3]
    c2 = -np.linalg.inv(Po2[:, 0:3]) @ Po2[:, 3]

    # new x axis(=direction of the baseline)
    v1 = (c1 - c2)
    # new y axes (orthogonal to new x and old z)
    v2 = np.cross(R1[2, :].T, v1)
    # new z axes (orthogonal to baseline and y)
    v3 = np.cross(v1, v2)

    # new extrinsic parameters
    R = np.array([v1.T / np.linalg.norm(v1),
                  v2.T / np.linalg.norm(v2),
                  v3.T / np.linalg.norm(v3)])
    # translation is left unchanged

    # new intrinsic parameters (arbitrary)
    A = (A1 + A2) / 2
    A[0, 1] = 0 # no skew

    # new projection matrices
    Pn1 = A @ np.append(R, np.array([-R @ c1]).T, axis=1)
    Pn2 = A @ np.append(R, np.array([-R @ c2]).T, axis=1)

    # rectifying image transformation
    T1 = Pn1[0:3, 0:3] @ np.linalg.inv(Po1[0:3, 0:3])
    T2 = Pn2[0:3, 0:3] @ np.linalg.inv(Po2[0:3, 0:3])

    return T1, T2, Pn1, Pn2


if __name__ == "__main__":
    img_shape = (768, 576)
    # Po1, Po2 = getPo()
    Po1 = np.array([[976.5, 53.82, -239.8, 387500],
                    [98.49, 933.3, 157.4, 242800],
                    [0.579, 0.1108, 0.8077, 1118]])
    Po2 = np.array([[976.7, 53.76, -240.0, 40030],
                    [98.68, 931.01, 156.71, 251700],
                    [0.5766, 0.11411, 0.8089, 1174]])
    T1, T2, Pn1, Pn2 = rectify(Po1, Po2)
    print("Po1:\n", Po1)
    print("Po2:\n", Po2)
    print("Pn1:\n", Pn1)
    print("Pn2:\n", Pn2)
    img1 = cv2.imread('Sport0_OG0.bmp')
    img2 = cv2.imread('Sport1_OG0.bmp')
    img1_warped = cv2.warpPerspective(img1, T1, img_shape, cv2.INTER_LANCZOS4)
    img2_warped = cv2.warpPerspective(img2, T2, img_shape, cv2.INTER_LANCZOS4)

    cv2.imshow('img1_origin', img1)
    cv2.imshow('img2_origin', img2)
    cv2.imshow('img1_rectified', img1_warped)
    cv2.imshow('img2_rectified', img2_warped)

    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break




