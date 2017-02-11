/*----------------------------------------------
 * Usage:
 * example_tracking_multitracker <video_name> [algorithm]
 *
 * example:
 * example_tracking_multitracker Bolt/img/%04d.jpg
 * example_tracking_multitracker faceocc2.webm KCF
 *--------------------------------------------------*/
#include<opencv/cv.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;
vector<Rect2d> rectToRect2d(vector<Rect> rectVec){
    vector<Rect2d> rect2dVec;
    for(int i=0;i<rectVec.size();i++){
        rect2dVec.push_back({rectVec[i].x, rectVec[i].y, rectVec[i].width, rectVec[i].height});
    }
    return rect2dVec;
}
int main( int argc, char** argv ){
    // show help
    if(argc<2){
        cout<<
            " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
                    " examples:\n"
                    " example_tracking_multitracker Bolt/img/%04d.jpg\n"
                    " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
            << endl;
        return 0;
    }

    // set the default tracking algorithm
    std::string trackingAlg = "KCF";

    HOGDescriptor hogDescriptor;
    hogDescriptor.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


    // set the tracking algorithm from parameter
    if(argc>2) {
        trackingAlg = argv[2];
    }

    // create the tracker
    //! [create]
    MultiTracker trackers(trackingAlg);
    //! [create]

    // set input video
    string video = argv[1];
    VideoCapture cap(video);
    Mat frame; cap>>frame;
    vector<Rect> objects;
    hogDescriptor.detectMultiScale(frame, objects, 0, Size(1,1), Size(0,0), 1.05, 2);
    cout<<objects.size()<<endl;
    trackers.add(frame, rectToRect2d(objects));

    //! [selectmulti]
    //! [init]

    int change=0;
    for ( ;; ){
        // get frame from the video
        cap >> frame;

        Rect2d addObject;
        if(change==0) {
            addObject=Rect2d(109,20,98,138);
            trackers.add(frame,addObject);
            change=1;
        }
        else if(change==1){
            addObject=Rect2d(29,40,98,138);
            trackers.add(frame,addObject);
            change=2;
        }


        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0)
            break;

        //update the tracking result
        //! [update]
        trackers.update(frame);
        //! [update]
        //cout<<trackers.objects.size()<<endl;
        for(unsigned i=0;i<trackers.objects.size();i++){
            rectangle( frame, trackers.objects[i], Scalar( 255, 0, 0 ), 2, 1 );
        }
        //! [result]

        // show image with the tracked object
        imshow("tracker",frame);

        //quit on ESC button
        if(waitKey(1)==27)break;
    }

}
