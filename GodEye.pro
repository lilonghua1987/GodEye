#-------------------------------------------------
#
# Project created by QtCreator 2015-02-06T13:58:08
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = GodEye
TEMPLATE = app


SOURCES += main.cpp\
        godeye.cpp \
    src/colorConversion.cpp \
    src/imageProcessing.cpp \
    src/random-standalone.cpp \
    src/recogonition.cpp \
    src/segmentation.cpp \
    src/smartOptimisation.cpp \
    src/SuperFormula.cpp \
    src/tools.cpp \
    qrecogonition.cpp

HEADERS  += godeye.h \
    src/colorConversion.h \
    src/imageProcessing.h \
    src/math_utils.h \
    src/random-standalone.h \
    src/recogonition.h \
    src/segmentation.h \
    src/smartOptimisation.h \
    src/SuperFormula.h \
    src/tools.h \
    qrecogonition.h

# Support c++11
QMAKE_CXXFLAGS += -std=c++11

# Configuration via pkg-config
CONFIG += link_pkgconfig

# Add the library needed
PKGCONFIG += opencv eigen3

LIBS +=\
    ../../lib/libekho.a\
    ../../lib/libFestival.a\
    ../../lib/libestools.a\
    ../../lib/libeststring.a\
    ../../lib/libestbase.a

LIBS+=-lpthread -lvorbisenc -lvorbis -lm -logg -lmp3lame -lsndfile -lncurses `pkg-config --libs libpulse-simple`


FORMS    += godeye.ui

RESOURCES += \
    resources.qrc
