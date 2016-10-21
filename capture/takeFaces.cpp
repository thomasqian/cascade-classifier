#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>

using namespace cv;

/** GLOBAL VARIABLES **/
String faceCascadeFile = "../data/lbpcascade_frontalface.xml";
String eyesCascadeFile = "../data/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier faceCascade;
CascadeClassifier eyesCascade;

int main(int argc, char **argv) {

    if (argc != 3) {
        std::cerr << "Usage: takeFaces outDir label" << std::endl;
        return -1;
    }

    VideoCapture cap(1);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera" << std::endl;
        return -1;
    }

    if (!faceCascade.load(faceCascadeFile)) {
        std::cerr << "Could not load face cascade file" << std::endl;
        return -1;
    }

    if (!eyesCascade.load(eyesCascadeFile)) {
        std::cerr << "Could not load eyes cascade file" << std::endl;
        return -1;
    }

    Mat img, gray;
    char x;
    std::string basedir = std::string(argv[1]) + "/s" + argv[2] + "/"; 
    if (!opendir(basedir.c_str())) {
        mkdir(basedir.c_str(), S_IRWXU | S_IRWXG);
    } 

    Mat faceROI, draw;
    std::vector<Rect> faces;
    Rect face;
    Point p1, p2;
    int edgeOffset = 10; // percent

    for (char i = '0'; i <= '9'; ++i) {
        std::cout << "Press space to take picture " << i << std::endl;

        while (true) {
            cap >> img;
            draw = img;

            // detect faces
            cvtColor(img, gray, CV_BGR2GRAY);
            //equalizeHist(gray, gray);

            faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(80, 80));

            if (!faces.size() == 0) {
                face = faces[0];
                
                face.x += face.width / edgeOffset;
                face.width -= face.width / edgeOffset;
                faceROI = gray(face);

                p1 = Point(face.x - 2, face.y - 2);
                p2 = Point(face.x + face.width + 2, face.y + face.height + 2);

                rectangle(draw, p1, p2, Scalar(255, 0, 0)); 
            }

            imshow("takeFaces", draw);
            if (waitKey(10) == 32 && faces.size() > 0) break;
        }
        destroyWindow("takeFaces");

        if (!faceROI.data) {
            std::cerr << "Image could not be taken" << std::endl;
            return -1;
        }

        Size cropSize(120, 120);
        resize(faceROI, faceROI, cropSize, 0, 0, INTER_LINEAR);

        std::string outdir = basedir + i + ".pgm";
        std::cout << "Writing to: " << outdir << std::endl;

        std::vector<int> compression_params; // Stores the compression parameters

        compression_params.push_back(CV_IMWRITE_PXM_BINARY); // Set to PXM compression
        compression_params.push_back(0); // Set type of PXM in our case PGM

        imwrite(outdir, faceROI, compression_params);
    }

    return 0;
}
