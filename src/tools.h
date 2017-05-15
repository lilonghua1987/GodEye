#pragma once

#include <string>
#include <string.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#ifndef PI
#define PI 3.14159265358979323846264338327950288419716939937510582
#endif

#ifdef _WIN32
#define ACCESS _access
#define MKDIR(a) _mkdir((a))
#elif _LINUX
#define ACCESS access
#define MKDIR(a) mkdir((a),0755)
#endif

using namespace std;

/*
计算运行时的时间
*/
class   RunTimer		
{
public: 
	void start();
	float stop();
	void timeDisplay(std::string disp);
	void fpsDisplay(std::string disp);
private:
	clock_t m_begin; 
	clock_t m_end;
};


/************************************************************************/
/*   一些工具方法                                                                   */
/************************************************************************/
class tools
{
public:
	tools(void);
	~tools(void);

	/************************************************************************/
	/* @prefix   文件名字                                                                   */
	/* @suffix 文件的存储格式        example:.jpg                                           */
	/************************************************************************/
	 static string fileNameFromTime(string prefix,string suffix);
	 static string fileNameFromTime(const char* prefix,const char* suffix);
	 static string fileNameFromTime(string path,string name,string suffix);
	 static string fileNameFromTime(const char* path,const char* name,const char* suffix);
	 static string fileName(string prefix,long w,long h,string suffix);
	 static string fileName(const char* prefix,long w,long h,const char* suffix);
	 static string fileName(string path,string name,long w,long h,string suffix);
	 static string fileName(const char* path,const char* name,long w,long h,const char* suffix);
     static string getFileName(const string& name);
     static string fileNamePart(const string& name);

	 static void getWords(std::vector<std::string>& words, const std::string& str, const char splite);

     static long bound(const long x, const long min, const long max)
	 {
		  return (x < min ? min : (x > max ? max : x));
     };

	 static long lowBound(const long x, const long min)
	 {
		 return (x < min) ? min:x;
	 }

	 static long upBound(const long x, const long max)
	 {
		 return (x > max) ? max:x;
	 }

	 template<typename T>
	 static T Max(const T &x, const T &y)
	 {
		  return (x < y ? y : x);
     };

	 template<typename T>
	 static T Min(const T &x, const T &y)
	 {
		  return (x > y ? y : x);
     };

	 template<typename T>
	 static int Round(const T& x)
	 {
		 return ((int)( (x) >= 0 ? (x) + .5 : (x) - .5));
	 };

	 static int CreatDir(const char * const pDir);

     static void  listFileByDir(const string& dir, const string& fileExtension, vector<string>& fileList);
};

