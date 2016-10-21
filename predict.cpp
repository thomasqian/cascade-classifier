#include "opencv2/core/core.hpp"
#include "opencv2/face.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

String faceCascadeFile = "data/lbpcascade_frontalface.xml";
CascadeClassifier faceCascade;

int main(int argc, const char *argv[]) {

    if (argc != 2) {
        cout << "usage: " << argv[0] << " model" << endl;
        exit(1);
    }

    VideoCapture cap(1);
    if (!cap.isOpened()) {
        cout << "Could not open camera" << endl;
        return -1;
    }

    if (!faceCascade.load(faceCascadeFile)) {
        std::cerr << "Could not load face cascade file" << std::endl;
        return -1;
    }

    Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer();
    model->setThreshold(85.0);
    model->load(argv[1]);

    Mat test, gray, faceROI;
    std::vector<Rect> faces;
    Rect face;
    Point p1, p2;
    int edgeOffset = 10; // percent

    while(true) {
        cout << "Press space to take picture" << endl;

        while(true) {
            cap >> test;

            // detect faces
            cvtColor(test, gray, CV_BGR2GRAY);
            equalizeHist(gray, gray);

            faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0, Size(80, 80));

            if (!faces.size() == 0) {
                face = faces[0];

                face.x += face.width / edgeOffset;
                face.width -= face.width / edgeOffset;
                faceROI = gray(face);

                p1 = Point(face.x - 2, face.y - 2);
                p2 = Point(face.x + face.width + 2, face.y + face.height + 2);

                rectangle(test, p1, p2, Scalar(255, 0, 0)); 
            }
            // end detect faces

            imshow("predict", test);
            if (waitKey(10) == 32 && faces.size() > 0) break;
            else if (waitKey(10) == 32 && faces.size() == 0) {
                std::cout << "No face detected" << std::endl;
            }
        }
        destroyWindow("predict");

        if (!faceROI.data) {
            cerr << "Image could not be taken" << endl;
            return -1;
        }

        int predictedLabel = -1;
        double predictedConfidence = 0.0;

        model->predict(faceROI, predictedLabel, predictedConfidence);

        if (predictedLabel == -1) {
            cout << "Unknown person" << endl;
        } else {
            cout << "Model predicted: Label: " << predictedLabel <<
                " ; Confidence: " << predictedConfidence << endl;
        }

        //cout << "Enter correct label:
    }

    return 0;
}
