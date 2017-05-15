#include "tools.h"

using namespace std;


void RunTimer::start()
{	
	m_begin=clock();
}


float RunTimer::stop()
{ 
	m_end=clock();
	return ( float(m_end-m_begin)/CLOCKS_PER_SEC );
}


void RunTimer::timeDisplay(std::string disp)
{ 
	std::cout << " Running time <" << disp << ">: " << stop() << " seconds." << std::endl;
}



void RunTimer::fpsDisplay(std::string disp)
{ 
	std::cout << " Running time <" << disp << ">: " << 1.0f/stop() << " frame per second." << std::endl;
}


//
tools::tools(void)
{
}


tools::~tools(void)
{
}


string tools::fileNameFromTime(string prefix,string suffix)
{
	string fileName;
	fileName.append(prefix);

	time_t	nowtime = time(NULL);
	struct	tm	*p;
	p = gmtime(&nowtime);
	char	filename[256] = {0};
	sprintf(filename,"%d_%d_%d_%d_%d_%d",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour+8,p->tm_min,p->tm_sec);
	fileName.append(filename);
	fileName.append(suffix);
	return fileName;
}


string tools::fileNameFromTime(const char* prefix,const char* suffix)
{
	string fileName;
	fileName.append(prefix);

	time_t	nowtime = time(NULL);
	struct	tm	*p;
	p = gmtime(&nowtime);
	char	filename[256] = {0};
	sprintf(filename,"%d_%d_%d_%d_%d_%d",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour+8,p->tm_min,p->tm_sec);
	fileName.append(filename);
	fileName.append(suffix);
	return fileName;
}


string tools::fileNameFromTime(string path,string name,string suffix)
{
	string fileName;
	if (!path.empty())
	{
		if ((path.find_last_of("\\") == path.length()-1) || (path.find_last_of("//") == path.length()-1)|| (path.find_last_of("/") == path.length()-1))
		{
			fileName.append(path);
		}else
		{
			fileName.append(path);
			fileName.append("\\");
		}		
	}

	if (!name.empty())
	{
		fileName.append(name);
	}

	if(suffix.empty()) return string();

	time_t	nowtime = time(NULL);
	struct	tm	*p;
	p = gmtime(&nowtime);
	char	filename[256] = {0};
	sprintf(filename,"%d_%d_%d_%d_%d_%d",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour+8,p->tm_min,p->tm_sec);
	fileName.append(filename);
	fileName.append(suffix);
	return fileName;
}


string tools::fileNameFromTime(const char* path,const char* name,const char* suffix)
{
	string fileName;
	if (path)
	{
        int iLen = strlen(path);
		if (path[iLen-1] == '\\' || path[iLen-1] == '/')
		{ 
			fileName.append(path);
		}else
		{
			fileName.append(path);
			fileName.append("\\");
		}
	}

	if (name)
	{
		fileName.append(name);
	}

	time_t	nowtime = time(NULL);
	struct	tm	*p;
	p = gmtime(&nowtime);
	char	filename[256] = {0};
	sprintf(filename,"%d_%d_%d_%d_%d_%d",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour+8,p->tm_min,p->tm_sec);
	fileName.append(filename);
	fileName.append(suffix);
	return fileName;
}


string tools::fileName(string prefix,long w,long h,string suffix)
{
	string fileName;
	fileName.append(prefix);
	fileName.append(to_string((long long )w));
	fileName.append("_");
	fileName.append(to_string((long long )h));
	fileName.append(suffix);
	return fileName;
}


string tools::fileName(const char* prefix,long w,long h,const char* suffix)
{
	string fileName;
	fileName.append(prefix);
	fileName.append(to_string((long long )w));
	fileName.append("_");
	fileName.append(to_string((long long )h));
	fileName.append(suffix);
	return fileName;
}


string tools::fileName(string path,string name,long w,long h,string suffix)
{
	string fileName;
	if (!path.empty())
	{
		if ((path.find_last_of("\\") == path.length()-1) || (path.find_last_of("//") == path.length()-1)|| (path.find_last_of("/") == path.length()-1))
		{
			fileName.append(path);
		}else
		{
			fileName.append(path);
			fileName.append("\\");
		}		
	}

	if (!name.empty())
	{
		fileName.append(name);
	}
	fileName.append(to_string((long long )w));
	fileName.append("_");
	fileName.append(to_string((long long )h));
	fileName.append(suffix);
	return fileName;
}


string tools::fileName(const char* path,const char* name,long w,long h,const char* suffix)
{
	string fileName;
	if (path)
	{
		int iLen = strlen(path);		
		if (path[iLen-1] == '\\' || path[iLen-1] == '/')
		{ 
			fileName.append(path);
		}else
		{
			fileName.append(path);
			fileName.append("\\");
		}
	}

	if (name)
	{
		fileName.append(name);
	}

	fileName.append(to_string((long long )w));
	fileName.append("_");
	fileName.append(to_string((long long )h));
	fileName.append(suffix);
	return fileName;
}


void tools::getWords(std::vector<std::string>& words, const std::string& str, const char splite)
{
	string word;
	for (int i = 0; i < str.length(); i++)
	{
		char temp = str.at(i);
		if ((i == (str.length() - 1)) && (temp != splite))
		{
			word += temp;
			temp = splite;
		}
		if (temp != splite)
		{
			word += temp;
			continue;
		}
		else
		{
			words.push_back(word);
            //word.swap(string());
            word = string();
		}
	}
}


string tools::getFileName(const string& name)
{
	if(name.empty()) return string();
	int pos = name.find_last_of("\\");
	if((pos+1) == name.length()) return string();

	return name.substr(pos+1,name.length());
}


string tools::fileNamePart(const string &name){
    if(name.empty())
        return string();

    int pos = name.find_last_of("/");

    if((pos+1) == name.length())
        return string();

    int ePos = name.find_last_of(".");

    if (ePos <= 0 || (ePos+1) == name.length()  || ePos <= pos){
        return string();
    }

    return name.substr(pos + 1, ePos - pos - 1);
}


int tools::CreatDir(const char * const pDir)
{
    int i = 0;
    int iRet = 0;
    int iLen;
    char* pszDir;

    if(NULL == pDir)
    {
        return 0;
    }

    pszDir = strdup(pDir);
    iLen = strlen(pDir);

    // 创建中间目录
    for (i = 0;i < iLen;i ++)
    {
        if (pszDir[i] == '\\' || pszDir[i] == '/')
        {
            pszDir[i] = '\0';

            //如果不存在,创建
            iRet = access(pszDir,F_OK);
            if (iRet != 0)
            {
                iRet = mkdir(pszDir,0755);
                if (iRet != 0)
                {
                    return -1;
                }
            }
            //支持linux,将所有\换成/
            pszDir[i] = '/';
        }
    }

    iRet = mkdir(pszDir,0755);
    free(pszDir);
    return iRet;
}


void  tools::listFileByDir(const string& dir, const string& fileExtension, vector<string>& fileList)
{
    string fileFolder = dir;

    if (!dir.empty())
    {
        if ( dir.find_last_of("/") <  (dir.length() - 1) )
        {
            fileFolder.append("/");
        }
    }else
    {
        fileFolder.append("/*");
    }


    // 遍历文件夹

    DIR *pDir;
    pDir = opendir(dir.c_str());

    struct dirent* fileInfo = NULL;    // 文件信息结构体

    // 1. 第一次查找
    if (NULL == pDir)
    {
        return;
    }

    // 2. 循环查找
    while ( ( fileInfo = readdir(pDir) ) != NULL )
    {
        if((strcmp(fileInfo->d_name,".")==0)||(strcmp(fileInfo->d_name,"..")==0))
        {
            continue;
        }else {
             string newPath = fileFolder + fileInfo->d_name;
            struct stat buf;
            lstat(newPath.c_str(),&buf);
             if( S_ISDIR(buf.st_mode) )
             {
                  listFileByDir(newPath, fileExtension,fileList);
             }else {
                 fileList.push_back(newPath);
             }
        }
    }
    closedir(pDir);
}
