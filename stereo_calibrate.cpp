#pragma warning( disable: 4996 )
/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warrenty, support or any guarentee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   OTHER OPENCV SITES:
   * The source code is on sourceforge at:
     http://sourceforge.net/projects/opencvlibrary/
   * The OpenCV wiki page (As of Oct 1, 2008 this is down for changing over servers, but should come back):
     http://opencvlibrary.sourceforge.net/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
   ************************************************** */

/*
	Modified by Martin Peris Martorell (info@martinperis.com) in order to accept some configuration
	parameters and store all the calibration data as xml files.

*/

/* Modified by Qi Yiru (Zoe) for the purpose of fulfilling FYP requirements.
For simplified installation process, this is the version without using CMake.*/

#include "opencv/cv.h"
#include "opencv/cxmisc.h"
#include "opencv/highgui.h"
#include "opencv/cvaux.h"
//#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
//#include <string>

#include <gl/glut.h>
#include <gl/glui.h>

using namespace std;


struct Point2D {
	GLfloat X, Y;
};

struct Point3D {
	GLfloat X, Y, Z;
};

vector<Point3D> points3D;
float Xmax, Xmin;

//
// Given a list of chessboard images, the number of corners (nx, ny)
// on the chessboards, and a flag: useCalibrated for calibrated (0) or
// uncalibrated (1: use cvStereoCalibrate(), 2: compute fundamental
// matrix separately) stereo. Calibrate the cameras and display the
// rectified results along with the computed disparity images.
//
static void
StereoCalib(const char* imageList, int nx, int ny, int useUncalibrated, float _squareSize)
{
    int displayCorners = 1;
    int showUndistorted = 1;
    bool isVerticalStereo = false;//OpenCV can handle left-right
                                      //or up-down camera arrangements
    const int maxScale = 1;
    const float squareSize = _squareSize; //Chessboard square size in cm
    FILE* f = fopen(imageList, "rt");
    int i, j, lr, nframes, n = nx*ny, N = 0; //n is no of corners
    vector<string> imageNames[2];
    vector<CvPoint3D32f> objectPoints;
    vector<CvPoint2D32f> points[2];
    vector<int> npoints;
    vector<uchar> active[2];
    vector<CvPoint2D32f> temp(n);
    //CvSize imageSize = cvSize(0,0);
	CvSize imageSize = {0,0};
    // ARRAY AND VECTOR STORAGE:
    double M1[3][3], M2[3][3], D1[5], D2[5];
    double R[3][3], T[3], E[3][3], F[3][3];
    double Q[4][4];
    CvMat _M1 = cvMat(3, 3, CV_64F, M1 );
    CvMat _M2 = cvMat(3, 3, CV_64F, M2 );
    CvMat _D1 = cvMat(1, 5, CV_64F, D1 );
    CvMat _D2 = cvMat(1, 5, CV_64F, D2 );
    CvMat _R = cvMat(3, 3, CV_64F, R );
    CvMat _T = cvMat(3, 1, CV_64F, T );
    CvMat _E = cvMat(3, 3, CV_64F, E );
    CvMat _F = cvMat(3, 3, CV_64F, F );
    CvMat _Q = cvMat(4,4, CV_64F, Q);
    if( displayCorners )
        cvNamedWindow( "corners", 1 );
// READ IN THE LIST OF CHESSBOARDS:
    if( !f )
    {
        fprintf(stderr, "can not open file %s\n", imageList );
        return;
    }
    for(i=0;;i++)
    {
        char buf[1024];
        int count = 0, result=0;
        lr = i % 2;
        vector<CvPoint2D32f>& pts = points[lr];
        if( !fgets( buf, sizeof(buf)-3, f ))
            break;
        size_t len = strlen(buf);
        while( len > 0 && isspace(buf[len-1]))
            buf[--len] = '\0';
        if( buf[0] == '#')
            continue;
        IplImage* img = cvLoadImage( buf, 0 );
        if( !img )
            break;
        imageSize = cvGetSize(img);
        imageNames[lr].push_back(buf);
    //FIND CHESSBOARDS AND CORNERS THEREIN:
        for( int s = 1; s <= maxScale; s++ )
        {
            IplImage* timg = img;
            if( s > 1 )
            {
                timg = cvCreateImage(cvSize(img->width*s,img->height*s),
                    img->depth, img->nChannels );
                cvResize( img, timg, CV_INTER_CUBIC ); //to enlarge an image (slow)
            }
            result = cvFindChessboardCorners( timg, cvSize(nx, ny),
                &temp[0], &count,
                CV_CALIB_CB_ADAPTIVE_THRESH |
                CV_CALIB_CB_NORMALIZE_IMAGE);
            if( timg != img )
                cvReleaseImage( &timg );
            if( result || s == maxScale )
                for( j = 0; j < count; j++ )
            {
                temp[j].x /= s;
                temp[j].y /= s;
            }
            if( result )
                break;
        }
        if( displayCorners )
        {
            printf("%s\n", buf);
            IplImage* cimg = cvCreateImage( imageSize, 8, 3 );
            cvCvtColor( img, cimg, CV_GRAY2BGR );
            cvDrawChessboardCorners( cimg, cvSize(nx, ny), &temp[0],
                count, result );
            cvShowImage( "corners", cimg );
            cvReleaseImage( &cimg );
            if( cvWaitKey(0) == 27 ) //Allow ESC to quit
                exit(-1);
        }
        else
            putchar('.');
        N = pts.size();
        pts.resize(N + n, cvPoint2D32f(0,0));
        active[lr].push_back((uchar)result);
    //assert( result != 0 );
        if( result )
        {
         //Calibration will suffer without subpixel interpolation
            cvFindCornerSubPix( img, &temp[0], count,
                cvSize(11, 11), cvSize(-1,-1),
                cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                30, 0.01) );
            copy( temp.begin(), temp.end(), pts.begin() + N );
        }
        cvReleaseImage( &img );
    }
    fclose(f);
    printf("\n");
// HARVEST CHESSBOARD 3D OBJECT POINT LIST:
    nframes = active[0].size();//Number of good chessboads found
    objectPoints.resize(nframes*n);
    for( i = 0; i < ny; i++ )
        for( j = 0; j < nx; j++ )
        objectPoints[i*nx + j] = cvPoint3D32f(i*squareSize, j*squareSize, 0);
    for( i = 1; i < nframes; i++ )
        copy( objectPoints.begin(), objectPoints.begin() + n,
        objectPoints.begin() + i*n );
    npoints.resize(nframes,n);
    N = nframes*n;
    CvMat _objectPoints = cvMat(1, N, CV_32FC3, &objectPoints[0] );
    CvMat _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    CvMat _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    CvMat _npoints = cvMat(1, npoints.size(), CV_32S, &npoints[0] );
    cvSetIdentity(&_M1);
    cvSetIdentity(&_M2);
    cvZero(&_D1);
    cvZero(&_D2);

// CALIBRATE THE STEREO CAMERAS
    printf("Running stereo calibration ...");
    fflush(stdout);
    cvStereoCalibrate( &_objectPoints, &_imagePoints1,
        &_imagePoints2, &_npoints,
        &_M1, &_D1, &_M2, &_D2,
        imageSize, &_R, &_T, &_E, &_F,
        cvTermCriteria(CV_TERMCRIT_ITER+
        CV_TERMCRIT_EPS, 100, 1e-5),
        CV_CALIB_FIX_ASPECT_RATIO +
        CV_CALIB_ZERO_TANGENT_DIST +
        CV_CALIB_SAME_FOCAL_LENGTH );
    printf(" done\n");
// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0
    vector<CvPoint3D32f> lines[2];
    points[0].resize(N);
    points[1].resize(N);
    _imagePoints1 = cvMat(1, N, CV_32FC2, &points[0][0] );
    _imagePoints2 = cvMat(1, N, CV_32FC2, &points[1][0] );
    lines[0].resize(N);
    lines[1].resize(N);
    CvMat _L1 = cvMat(1, N, CV_32FC3, &lines[0][0]);
    CvMat _L2 = cvMat(1, N, CV_32FC3, &lines[1][0]);
//Always work in undistorted space
    cvUndistortPoints( &_imagePoints1, &_imagePoints1,
        &_M1, &_D1, 0, &_M1 );
    cvUndistortPoints( &_imagePoints2, &_imagePoints2,
        &_M2, &_D2, 0, &_M2 );
    cvComputeCorrespondEpilines( &_imagePoints1, 1, &_F, &_L1 );
    cvComputeCorrespondEpilines( &_imagePoints2, 2, &_F, &_L2 );
    double avgErr = 0;
    for( i = 0; i < N; i++ )
    {
        double err = fabs(points[0][i].x*lines[1][i].x +
            points[0][i].y*lines[1][i].y + lines[1][i].z)
            + fabs(points[1][i].x*lines[0][i].x +
            points[1][i].y*lines[0][i].y + lines[0][i].z);
        avgErr += err;
    }
    printf( "avg err = %g\n", avgErr/(nframes*n) );
//COMPUTE AND DISPLAY RECTIFICATION
    if( showUndistorted )
    {
        CvMat* mx1 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* my1 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* mx2 = cvCreateMat( imageSize.height,

            imageSize.width, CV_32F );
        CvMat* my2 = cvCreateMat( imageSize.height,
            imageSize.width, CV_32F );
        CvMat* img1r = cvCreateMat( imageSize.height,
            imageSize.width, CV_8U );
        CvMat* img2r = cvCreateMat( imageSize.height,
            imageSize.width, CV_8U );
        CvMat* disp = cvCreateMat( imageSize.height,
            imageSize.width, CV_16S );
        CvMat* vdisp = cvCreateMat( imageSize.height,
            imageSize.width, CV_8U );
        CvMat* pair;
        double R1[3][3], R2[3][3], P1[3][4], P2[3][4];
        CvMat _R1 = cvMat(3, 3, CV_64F, R1);
        CvMat _R2 = cvMat(3, 3, CV_64F, R2);
// IF BY CALIBRATED (BOUGUET'S METHOD)
        if( useUncalibrated == 0 )
        {
            CvMat _P1 = cvMat(3, 4, CV_64F, P1);
            CvMat _P2 = cvMat(3, 4, CV_64F, P2);
            cvStereoRectify( &_M1, &_M2, &_D1, &_D2, imageSize,
                &_R, &_T,
                &_R1, &_R2, &_P1, &_P2, &_Q,
                0/*CV_CALIB_ZERO_DISPARITY*/ );
            isVerticalStereo = fabs(P2[1][3]) > fabs(P2[0][3]);
    //Precompute maps for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_P1,mx1,my1);
            cvInitUndistortRectifyMap(&_M2,&_D2,&_R2,&_P2,mx2,my2);
            
    //Save parameters
            cvSave("M1.xml",&_M1);
            cvSave("D1.xml",&_D1);
            cvSave("R1.xml",&_R1);
            cvSave("P1.xml",&_P1);
            cvSave("M2.xml",&_M2);
            cvSave("D2.xml",&_D2);
            cvSave("R2.xml",&_R2);
            cvSave("P2.xml",&_P2);
            cvSave("Q.xml",&_Q);
            cvSave("mx1.xml",mx1);
            cvSave("my1.xml",my1);
            cvSave("mx2.xml",mx2);
            cvSave("my2.xml",my2);

        }
//OR ELSE HARTLEY'S METHOD
        else if( useUncalibrated == 1 || useUncalibrated == 2 )
     // use intrinsic parameters of each camera, but
     // compute the rectification transformation directly
     // from the fundamental matrix
        {
            double H1[3][3], H2[3][3], iM[3][3];
            CvMat _H1 = cvMat(3, 3, CV_64F, H1);
            CvMat _H2 = cvMat(3, 3, CV_64F, H2);
            CvMat _iM = cvMat(3, 3, CV_64F, iM);
    //Just to show you could have independently used F
            if( useUncalibrated == 2 )
                cvFindFundamentalMat( &_imagePoints1,
                &_imagePoints2, &_F);
            cvStereoRectifyUncalibrated( &_imagePoints1,
                &_imagePoints2, &_F,
                imageSize,
                &_H1, &_H2, 3);
            cvInvert(&_M1, &_iM);
            cvMatMul(&_H1, &_M1, &_R1);
            cvMatMul(&_iM, &_R1, &_R1);
            cvInvert(&_M2, &_iM);
            cvMatMul(&_H2, &_M2, &_R2);
            cvMatMul(&_iM, &_R2, &_R2);
    //Precompute map for cvRemap()
            cvInitUndistortRectifyMap(&_M1,&_D1,&_R1,&_M1,mx1,my1);

            cvInitUndistortRectifyMap(&_M2,&_D1,&_R2,&_M2,mx2,my2);
        }
        else
            assert(0);
        cvNamedWindow( "rectified", 1 );
// RECTIFY THE IMAGES AND FIND DISPARITY MAPS
        if( !isVerticalStereo )
            pair = cvCreateMat( imageSize.height, imageSize.width*2,
            CV_8UC3 );
        else
            pair = cvCreateMat( imageSize.height*2, imageSize.width,
            CV_8UC3 );
//Setup for finding stereo corrrespondences
        CvStereoBMState *BMState = cvCreateStereoBMState();
        assert(BMState != 0);
        BMState->preFilterSize=41;
        BMState->preFilterCap=31;
        BMState->SADWindowSize=41;
        BMState->minDisparity=-64;
        BMState->numberOfDisparities=128;
        BMState->textureThreshold=10;
        BMState->uniquenessRatio=15;
        for( i = 0; i < nframes; i++ )
        {
            IplImage* img1=cvLoadImage(imageNames[0][i].c_str(),0);
            IplImage* img2=cvLoadImage(imageNames[1][i].c_str(),0);
            if( img1 && img2 )
            {
                CvMat part;
                cvRemap( img1, img1r, mx1, my1 );
                cvRemap( img2, img2r, mx2, my2 );
                if( !isVerticalStereo || useUncalibrated != 0 )
                {
              // When the stereo camera is oriented vertically,
              // useUncalibrated==0 does not transpose the
              // image, so the epipolar lines in the rectified
              // images are vertical. Stereo correspondence
              // function does not support such a case.
                    cvFindStereoCorrespondenceBM( img1r, img2r, disp,
                        BMState);
                    cvNormalize( disp, vdisp, 0, 256, CV_MINMAX );
                    cvNamedWindow( "disparity" );
                    cvShowImage( "disparity", vdisp );
                }
                if( !isVerticalStereo )
                {
                    cvGetCols( pair, &part, 0, imageSize.width );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetCols( pair, &part, imageSize.width,
                        imageSize.width*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < imageSize.height; j += 16 )
                        cvLine( pair, cvPoint(0,j),
                        cvPoint(imageSize.width*2,j),
                        CV_RGB(0,255,0));
                }
                else
                {
                    cvGetRows( pair, &part, 0, imageSize.height );
                    cvCvtColor( img1r, &part, CV_GRAY2BGR );
                    cvGetRows( pair, &part, imageSize.height,
                        imageSize.height*2 );
                    cvCvtColor( img2r, &part, CV_GRAY2BGR );
                    for( j = 0; j < imageSize.width; j += 16 )
                        cvLine( pair, cvPoint(j,0),
                        cvPoint(j,imageSize.height*2),
                        CV_RGB(0,255,0));
                }
                cvShowImage( "rectified", pair );
                if( cvWaitKey() == 27 )
                    break;
            }
            cvReleaseImage( &img1 );
            cvReleaseImage( &img2 );
        }
        cvReleaseStereoBMState(&BMState);
        cvReleaseMat( &mx1 );
        cvReleaseMat( &my1 );
        cvReleaseMat( &mx2 );
        cvReleaseMat( &my2 );
        cvReleaseMat( &img1r );
        cvReleaseMat( &img2r );
        cvReleaseMat( &disp );
    }
}

void init ()//checked
{	
	//  Set the frame buffer clear color to black. 
	glEnable(GL_DEPTH_TEST); 
	 
	glClearColor (0.8, 0.8, 1, 0.0);
	GLfloat global_ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient);
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);


	printf("Starting drawing 3D corners...\n");
}

void drawPoints(vector<Point3D> points3D){

	//float fit_scale = 1.0/(Xmax - Xmin);
	//fit_x = -(model.max_x+model.min_x)/2.0;
	//fit_y = -(model.max_y+model.min_y)/2.0;
	//fit_z = -(model.max_z+model.min_z)/2.0;
	

	//std::cout<<" within draw_mesh: "<<primitive<<std::endl;

	//normalize the model
	//glEnable(GL_NORMALIZE);  //ensure the length of normals is 1
	//glScalef (fit_scale, fit_scale, fit_scale);

	glDisable( GL_LIGHTING );
	glBegin(GL_LINES);
			glColor3f(0, 0, 1);
			//glVertex3f(0, 10, 10);
			//glVertex3f(10, 0, 10);
			glVertex3f(points3D[0].X,points3D[0].Y,0);	
			glVertex3f(points3D[8].X,points3D[8].Y,0);
			std::cout<<"Drawing X: "<<points3D[0].X<<std::endl;
				std::cout<<"Drawing Y: "<<points3D[0].Y<<std::endl;
				std::cout<<"Drawing Z: "<<points3D[0].Z<<std::endl;
	//glColor3f(0.329412, 0.329412, 0.329412);
			//std::cout<<" Size of points "<<sizeof(points3D)<<std::endl;
			/*for(int x=0;x<8;x++){
				glVertex3f(points3D[x].X/20,points3D[x].Y/20,points3D[x].Z/20);	
				glVertex3f(points3D[x+1].X/20,points3D[x+1].Y/20,points3D[x+1].Z/20);	
				std::cout<<"Drawing X: "<<points3D[x].X<<std::endl;
				std::cout<<"Drawing Y: "<<points3D[x].Y<<std::endl;
				std::cout<<"Drawing Z: "<<points3D[x].Z<<std::endl;
			}*/
			
	glEnd();
}

void draw_axis(void)//checked
{
	float length=50;
	GLUquadricObj *quadratic;
	quadratic = gluNewQuadric();
	gluCylinder(quadratic,0.3f,0.3f,length,50,50);//draw the axis line
	glTranslatef(0.0f, 0.0f, length); 
	glutSolidCone(0.6f,2.0f,50,50); //draw the axis arrow
}

//x,y,z-axis
void draw_system(void)//checked
{   
	
	//glPushMatrix();
		
		glShadeModel(GL_SMOOTH);
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

		 glPushMatrix();
		glColor3f(0.1, 1.0, 0.1);	 //green z-axis
			draw_axis();
		glPopMatrix();

		glPushMatrix();
			 glColor3f(0.1, 0.1, 1.0);//blue y-axis
			glRotatef(-90.0f, 1.0f, 0.0f, 0.0f); 
		draw_axis();
		glPopMatrix();
	
		glPushMatrix();
			glColor3f(1.0, 0.1, 0.1); //red x-axis
			glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
			draw_axis();
		glPopMatrix();

		
	//glPopMatrix();

}
void draw_grid(void)//checked
{
	//glPushMatrix();
		//draw x-y plane
		glDisable( GL_LIGHTING );
		glBegin(GL_LINES);
			glColor3f(0.329412, 0.329412, 0.329412);

			for(int x=-10;x<=10;x++){
				if(x!=0){
					glVertex3f(10*x,-100.0,0.0);
					glVertex3f(10*x,100.0,0.0);	
				}
			}
			for(int y=-10;y<=10;y++){
				if(y!=0){
					glVertex3f(-100.0,10*y,0.0);
					glVertex3f(100.0,10*y,0.0);
				}
			}
		glEnd();

		//draw lines along the axes
		glBegin(GL_LINES);
			glColor3f(1, 1, 1);
			glVertex3f(-100.0,0.0,0.0);
			glVertex3f(100.0,0.0,0.0);			
			glVertex3f(0.0,-100.0,0.0);
			glVertex3f(0.0,100.0,0.0);
			
		glEnd();
	//glPopMatrix();
}

void disp(void){

	// Clear color and depth buffers
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	// setup the perspective projection
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	//if(perspective)
	//{
		//Perspective perjection by default
		gluPerspective(45.0, 1, 0.1, 400.0);
		
		
	//}else
	//{
		//Orothographic porjection
		/*if (window_width <= window_height) {
			glOrtho(-1.5, 1.5, -1.5 / aspect, 1.5 / aspect, 0.1, 100.0);  // aspect <= 1
		} else {*/
			//glOrtho(-1.5 * aspect, 1.5 * aspect, -1.5, 1.5, 0.1, 100.0);  // aspect > 1
		// }		
		
	//}


	glMatrixMode(GL_MODELVIEW);   //To operate on model-view matrix
	glLoadIdentity();   //Reset the model-view matrix
	gluLookAt(-100,-100,20,0,0,0,0,0,1); 

/*
	switch(camera)
	{
		case TD:
			gluLookAt(4,4,4,0,0,0,0,0,1); 
			break;
		case X:
			//y-z plane
			
			gluLookAt(4,0,0,0,0,0,0,1,0); 
			break;
		case Y:
			//z-x plane
			
			gluLookAt(0,4,0,0,0,0,0,0,-1); 
			break;

		case Z:
			//x-y plane
			
			gluLookAt(0,0,4,0,0,0,0,1,0); 
			break;
	}
	
	*/
	// rotate and scale the object
	//glRotatef(x_angle, 0, 1,0); 
	//glRotatef(y_angle, 1,0,0); 
	//glScalef(scale_size, scale_size, scale_size); 

	//  Apply the translation
			//glTranslatef (translate_xy[0], translate_xy[1], -translate_z);
			//glTranslatef(xmove, ymove, 0.0);

	/*
			//  Apply the translation
			glTranslatef (translate_xy[0], translate_xy[1], -translate_z);
			glTranslatef(xmove, ymove, 0.0);


			//  Apply the scaling
			glScalef (scale, scale, scale);

		
		
			//  Apply the rotation matrix
			glMultMatrixf (rotation_matrix);

			// rotate and scale the object based on mouse
			glRotatef(xrotate, 1,0,0); 
			glRotatef(yrotate, 0,1,0); 
			glRotatef(zrotate, 0,0,1);
			*/
	
			//draw x-y plane
			//if(grid) 
				draw_grid(); 
			//draw coordinate system
			//if(axis) 
				draw_system();
			//draw model
			//if(object_type!=0) 
				//draw_object();
	
		drawPoints(points3D);
  
	// swap the buffers
	glutSwapBuffers(); 

}

int main(int argc, char *argv[])
{
    int nx = 9, ny = 6;
    float squareSize = 2.5;
	const char* imageList;
	imageList = "list.txt";
    int fail = 0;
    //Check command line
	/*std::cout<<" No Argument Count: "<<argc<<std::endl;
    if (argc != 5)
    {
        fprintf(stderr,"USAGE: %s imageList nx ny squareSize\n",argv[0]);
        fprintf(stderr,"\t imageList : Filename of the image list (string). Example : list.txt\n");
        fprintf(stderr,"\t nx : Number of horizontal squares (int > 0). Example : 9\n");
        fprintf(stderr,"\t ny : Number of vertical squares (int > 0). Example : 6\n");
        fprintf(stderr,"\t squareSize : Size of a square (float > 0). Example : 2.5\n");
        system("PAUSE");
		return 1;
    } 

		nx = atoi(argv[2]);
    ny = atoi(argv[3]);
    squareSize = (float)atof(argv[4]);

    if (nx <= 0)
    {
        fail = 1;
        fprintf(stderr, "ERROR: nx value can not be <= 0\n");
    }
    if (ny <= 0)
    {
        fail = 1;
        fprintf(stderr, "ERROR: ny value can not be <= 0\n");
    }   
    if (squareSize <= 0.0)
    {
        fail = 1;
        fprintf(stderr, "ERROR: squareSize value can not be <= 0\n");
    }   
	
    if(fail != 0) {
		system("PAUSE");
		return 1;
		//system("PAUSE");
	}*/
    //StereoCalib(argv[1], nx, ny, 0, squareSize);
	StereoCalib(imageList, nx, ny, 0, squareSize);
	
	CvSize imageSizeL = {0,0};
	CvSize imageSizeR = {0,0};

	CvMat *Q = (CvMat *)cvLoad("Q.xml",NULL,NULL,NULL);
	CvMat *mx1 = (CvMat *)cvLoad("mx1.xml",NULL,NULL,NULL);
	CvMat *my1 = (CvMat *)cvLoad("my1.xml",NULL,NULL,NULL);
	CvMat *mx2 = (CvMat *)cvLoad("mx2.xml",NULL,NULL,NULL);
	CvMat *my2 = (CvMat *)cvLoad("my2.xml",NULL,NULL,NULL);


	IplImage* imgLeftOrig=cvLoadImage("imagenes/left01.ppm",0);
    IplImage* imgRightOrig=cvLoadImage("imagenes/right01.ppm",0);

	imageSizeL = cvGetSize(imgLeftOrig);
	imageSizeR = cvGetSize(imgRightOrig);

	CvMat* imgLeftUndistorted = cvCreateMat( imageSizeL.height,
            imageSizeL.width, CV_8U );
    CvMat* imgRightUndistoreted = cvCreateMat( imageSizeR.height,
            imageSizeR.width, CV_8U );

	cvRemap(imgLeftOrig, imgLeftUndistorted, mx1, my1);
	cvRemap(imgRightOrig, imgRightUndistoreted, mx2, my2);

	//vector<Point2D> points2DL;
	//vector<Point2D> points2DR;
	

	float coordinatesL[9][2] = {{2.07,4.87},{2.64,4.85},{3.21,4.83},{3.82,4.80},{4.72,4.79},{5.04,4.77},{5.65,4.75},{6.27,4.74},{6.89,4.72}};
	float coordinatesR[9][2] = {{0.76,5.04},{1.28,5.02},{1.82,4.99},{2.38,4.99},{2.95,4.96},{3.56,4.94},{4.17,4.91},{4.80,4.90},{5.43,4.89}};

	for(int i=0; i<9; i++){
		Point2D l, r;
		Point3D p;
		l.X = coordinatesL[i][0];
		l.Y = coordinatesL[i][1];

		r.X = coordinatesR[i][0];
		r.Y = coordinatesR[i][1];

		float d = r.X - l.X;

		float X = l.X * cvmGet(Q,0,0) + cvmGet(Q,0,3);
		float Y = l.Y * cvmGet(Q,1,1) + cvmGet(Q,1,3);
		float Z = cvmGet(Q,2,3);
		float W = d * cvmGet(Q,3,2) + cvmGet(Q,3,3);

		p.X = X / W;
		p.Y = Y / W;
		p.Z = Z / W;

		points3D.push_back(p);

		std::cout<<" World coordinates: Point "<<i<<std::endl;
		std::cout<<" X: "<<points3D[i].X<<std::endl;
		std::cout<<" Y: "<<points3D[i].Y<<std::endl;
		std::cout<<" Z: "<<points3D[i].Z<<std::endl;
		

		if (i==0){
			Xmax = p.X;
			Xmin = p.X;
		}
		else{
			if(p.X>Xmax)
				Xmax = p.X;
			if(p.X<Xmin)
				Xmin = p.X;
		}
		
	}
	
	 // normal initialisation
	/*glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(500,500);
	glutInitWindowPosition(100,100);
	static int win;
	win = glutCreateWindow("3D corners");

	init();

	glutDisplayFunc(disp);

	*/

/*	std::cout<<" World coordinates: "<<std::endl;
	std::cout<<" X: "<<X<<std::endl;
	std::cout<<" Y: "<<Y<<std::endl;
	std::cout<<" Z: "<<Z<<std::endl;
*/	
	system("PAUSE");
	glutMainLoop();
	system("PAUSE");
    return 0;
	
}
