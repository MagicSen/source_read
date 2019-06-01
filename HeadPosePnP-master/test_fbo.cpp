#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "freeglut.lib")

#include <stdio.h>
#include <gl/glew.h>
#include <gl/glut.h>
#include "GL/freeglut.h"
#include <iostream>
#include <vector>

typedef unsigned char           uint8_t;

int windowWidth = 0;
int windowHeight = 0;

const int TexWidth = 512;
const int TexHeight = 512;

bool leftMouseDown = false;
float mouseX, mouseY;
float cameraAngleX, cameraAngleY;
float xRot, yRot;

GLuint textureID;
GLuint frameBufferID;
GLuint renderBufferID;

void drawCube()
{
	glBindTexture(GL_TEXTURE_2D, textureID);
	glColor4f(1, 1, 1, 1);

	glBegin(GL_QUADS);
	//Front
	glNormal3d(0, 0, 1);
	glVertex3d(-1, -1, 1);   glTexCoord2d(0, 0);
	glVertex3d(1, -1, 1);   glTexCoord2d(1, 0);
	glVertex3d(1, 1, 1);    glTexCoord2d(1, 1);
	glVertex3d(-1, 1, 1);   glTexCoord2d(0, 1);

	//Back
	glNormal3d(0, 0, -1);
	glVertex3d(1, -1, -1); glTexCoord2d(0, 0);
	glVertex3d(-1, -1, -1); glTexCoord2d(1, 0);
	glVertex3d(-1, 1, -1);  glTexCoord2d(1, 1);
	glVertex3d(1, 1, -1);   glTexCoord2d(0, 1);

	//Left
	glNormal3d(-1, 0, 0);
	glVertex3d(-1, -1, -1); glTexCoord2d(0, 0);
	glVertex3d(-1, -1, 1);  glTexCoord2d(1, 0);
	glVertex3d(-1, 1, 1);   glTexCoord2d(1, 1);
	glVertex3d(-1, 1, -1);  glTexCoord2d(0, 1);

	//Right
	glNormal3d(1, 0, 0);
	glVertex3d(1, -1, 1);   glTexCoord2d(0, 0);
	glVertex3d(1, -1, -1);  glTexCoord2d(1, 0);
	glVertex3d(1, 1, -1);   glTexCoord2d(1, 1);
	glVertex3d(1, 1, 1);    glTexCoord2d(0, 1);

	//Top
	glNormal3d(0, 1, 0);
	glVertex3d(-1, 1, 1);   glTexCoord2d(0, 0);
	glVertex3d(1, 1, 1);    glTexCoord2d(1, 0);
	glVertex3d(1, 1, -1);   glTexCoord2d(1, 1);
	glVertex3d(-1, 1, -1);  glTexCoord2d(0, 1);

	//Bottom
	glNormal3d(0, -1, 0);
	glVertex3d(1, -1, 1);   glTexCoord2d(0, 0);
	glVertex3d(-1, -1, 1);  glTexCoord2d(1, 0);
	glVertex3d(-1, -1, -1); glTexCoord2d(1, 1);
	glVertex3d(1, -1, -1);  glTexCoord2d(0, 1);

	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
}


void ChangeSize(int w, int h)
{
	windowWidth = w;
	windowHeight = h;

	if (h == 0)
		h = 1;
}

void SetupRC()
{
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TexWidth, TexHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glGenRenderbuffers(1, &renderBufferID);
	glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, TexWidth, TexHeight);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glGenFramebuffers(1, &frameBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		fprintf(stderr, "GLEW Error: %s\n", "FRAME BUFFER STATUS Error!");
		return;
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
}

void RenderScene(void)
{
	//设置渲染到纹理的视口和投影矩阵
	glViewport(0, 0, TexWidth, TexHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (float)(TexWidth) / TexHeight, 1.0f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//渲染到纹理
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor3f(1, 0, 1);
	glPushMatrix();
	glTranslated(0, 0.0, -5);
	glRotated(xRot, 1, 0, 0);
	glRotated(yRot, 0, 1, 0);
	glutSolidTeapot(1.0);
	glPopMatrix();

	//切换到窗口系统的帧缓冲区
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glViewport(0, 0, windowWidth, windowHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (float)(windowWidth) / windowHeight, 1.0f, 100.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glTranslated(0, 0, -5);
	glRotated(cameraAngleY*0.5, 1, 0, 0);
	glRotated(cameraAngleX*0.5, 0, 1, 0);
	glColor3d(1.0, 1.0, 1.0);
	drawCube();
	glutSwapBuffers();
}


void MouseFuncCB(int button, int state, int x, int y)
{
	mouseX = x;
	mouseY = y;

	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			leftMouseDown = true;
		}
		else if (state == GLUT_UP)
		{
			leftMouseDown = false;
		}
	}

}


void MouseMotionFuncCB(int x, int y)
{
	if (leftMouseDown)
	{
		cameraAngleX += (x - mouseX);
		cameraAngleY += (y - mouseY);

		mouseX = x;
		mouseY = y;
	}

	glutPostRedisplay();
}


void TimerFuncCB(int value)
{
	xRot += 2;
	yRot += 3;
	glutPostRedisplay();
	glutTimerFunc(33, TimerFuncCB, 1);
}


int main(int argc, char* argv[])
{
	//std::cout << argc << " " << argv[0] << std::endl;
	//char exe_name[] = "testsss";
	//char **exe_name = NULL;
	//glutInit(&argc, argv);
	int c = 1;
	char *exe_name[1] = { (char*)"Nothing" };
	glutInit(&c, exe_name);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutCreateWindow("OpenGL");
	glutReshapeFunc(ChangeSize);
	glutDisplayFunc(RenderScene);
	glutMouseFunc(MouseFuncCB);
	glutMotionFunc(MouseMotionFuncCB);
	glutTimerFunc(33, TimerFuncCB, 1);
	GLenum err = glewInit();

	if (GLEW_OK != err) {
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		system("pause");
		return 1;
	}

	SetupRC();

	//glutMainLoopEvent();
	glutMainLoop();
	system("pause");

	////glutMainLoop();


	return 0;
}