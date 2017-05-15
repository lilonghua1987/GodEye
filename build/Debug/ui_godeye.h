/********************************************************************************
** Form generated from reading UI file 'godeye.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GODEYE_H
#define UI_GODEYE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GodEye
{
public:
    QWidget *centralWidget;
    QLabel *srcImgLabel;
    QLabel *resultImgLable;
    QPushButton *runButton;
    QGroupBox *featureBox;
    QRadioButton *sift;
    QRadioButton *surf;
    QRadioButton *brief;
    QRadioButton *brisk;
    QRadioButton *orb;
    QRadioButton *freak;
    QGroupBox *matchBox;
    QRadioButton *flann;
    QRadioButton *brute;
    QRadioButton *brute1;
    QRadioButton *bruteH1;
    QRadioButton *bruteH2;
    QPushButton *imgButton;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *GodEye)
    {
        if (GodEye->objectName().isEmpty())
            GodEye->setObjectName(QStringLiteral("GodEye"));
        GodEye->resize(1080, 680);
        QIcon icon;
        icon.addFile(QStringLiteral(":/form/mainwin/resources/20150207051709200_easyicon_net_32.ico"), QSize(), QIcon::Normal, QIcon::Off);
        GodEye->setWindowIcon(icon);
        GodEye->setStyleSheet(QStringLiteral("background-image: url(:/form/mainwin/resources/bg.jpg);"));
        centralWidget = new QWidget(GodEye);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        srcImgLabel = new QLabel(centralWidget);
        srcImgLabel->setObjectName(QStringLiteral("srcImgLabel"));
        srcImgLabel->setGeometry(QRect(0, 0, 472, 657));
        resultImgLable = new QLabel(centralWidget);
        resultImgLable->setObjectName(QStringLiteral("resultImgLable"));
        resultImgLable->setGeometry(QRect(472, 0, 472, 657));
        runButton = new QPushButton(centralWidget);
        runButton->setObjectName(QStringLiteral("runButton"));
        runButton->setGeometry(QRect(960, 600, 100, 27));
        QIcon icon1;
        icon1.addFile(QStringLiteral(":/form/mainwin/resources/20150207055222880_easyicon_net_64.ico"), QSize(), QIcon::Normal, QIcon::Off);
        runButton->setIcon(icon1);
        runButton->setIconSize(QSize(26, 26));
        featureBox = new QGroupBox(centralWidget);
        featureBox->setObjectName(QStringLiteral("featureBox"));
        featureBox->setGeometry(QRect(944, 0, 136, 190));
        featureBox->setStyleSheet(QLatin1String("background-color: qlineargradient(spread:pad, x1:0, y1:0.948864, x2:1, y2:0, stop:0 rgba(67, 128, 0, 255), stop:1 rgba(255, 255, 255, 255));\n"
""));
        sift = new QRadioButton(featureBox);
        sift->setObjectName(QStringLiteral("sift"));
        sift->setGeometry(QRect(0, 20, 120, 23));
        sift->setChecked(true);
        surf = new QRadioButton(featureBox);
        surf->setObjectName(QStringLiteral("surf"));
        surf->setGeometry(QRect(0, 50, 120, 23));
        brief = new QRadioButton(featureBox);
        brief->setObjectName(QStringLiteral("brief"));
        brief->setGeometry(QRect(0, 80, 120, 23));
        brisk = new QRadioButton(featureBox);
        brisk->setObjectName(QStringLiteral("brisk"));
        brisk->setGeometry(QRect(0, 110, 120, 23));
        orb = new QRadioButton(featureBox);
        orb->setObjectName(QStringLiteral("orb"));
        orb->setGeometry(QRect(0, 140, 120, 23));
        freak = new QRadioButton(featureBox);
        freak->setObjectName(QStringLiteral("freak"));
        freak->setGeometry(QRect(0, 170, 120, 23));
        matchBox = new QGroupBox(centralWidget);
        matchBox->setObjectName(QStringLiteral("matchBox"));
        matchBox->setGeometry(QRect(944, 200, 136, 180));
        flann = new QRadioButton(matchBox);
        flann->setObjectName(QStringLiteral("flann"));
        flann->setGeometry(QRect(0, 20, 120, 23));
        flann->setChecked(true);
        brute = new QRadioButton(matchBox);
        brute->setObjectName(QStringLiteral("brute"));
        brute->setGeometry(QRect(0, 50, 120, 23));
        brute1 = new QRadioButton(matchBox);
        brute1->setObjectName(QStringLiteral("brute1"));
        brute1->setGeometry(QRect(0, 80, 120, 23));
        bruteH1 = new QRadioButton(matchBox);
        bruteH1->setObjectName(QStringLiteral("bruteH1"));
        bruteH1->setGeometry(QRect(0, 110, 120, 23));
        bruteH2 = new QRadioButton(matchBox);
        bruteH2->setObjectName(QStringLiteral("bruteH2"));
        bruteH2->setGeometry(QRect(0, 140, 120, 23));
        imgButton = new QPushButton(centralWidget);
        imgButton->setObjectName(QStringLiteral("imgButton"));
        imgButton->setGeometry(QRect(960, 460, 95, 27));
        GodEye->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(GodEye);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        GodEye->setStatusBar(statusBar);

        retranslateUi(GodEye);

        QMetaObject::connectSlotsByName(GodEye);
    } // setupUi

    void retranslateUi(QMainWindow *GodEye)
    {
        GodEye->setWindowTitle(QApplication::translate("GodEye", "GodEye", 0));
        srcImgLabel->setText(QString());
        resultImgLable->setText(QString());
        runButton->setText(QApplication::translate("GodEye", "Run", 0));
        featureBox->setTitle(QApplication::translate("GodEye", "Fearture", 0));
        sift->setText(QApplication::translate("GodEye", "SIFT", 0));
        surf->setText(QApplication::translate("GodEye", "SURF", 0));
        brief->setText(QApplication::translate("GodEye", "BRIEF", 0));
        brisk->setText(QApplication::translate("GodEye", "BRISK", 0));
        orb->setText(QApplication::translate("GodEye", "ORB", 0));
        freak->setText(QApplication::translate("GodEye", "FREAK", 0));
        matchBox->setTitle(QApplication::translate("GodEye", "Match", 0));
        flann->setText(QApplication::translate("GodEye", "FlannBased", 0));
        brute->setText(QApplication::translate("GodEye", "BruteForce", 0));
        brute1->setText(QApplication::translate("GodEye", "BruteForceL1", 0));
        bruteH1->setText(QApplication::translate("GodEye", "BruteFH1", 0));
        bruteH2->setText(QApplication::translate("GodEye", "BruteFH2", 0));
        imgButton->setText(QApplication::translate("GodEye", "OpenImage", 0));
    } // retranslateUi

};

namespace Ui {
    class GodEye: public Ui_GodEye {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GODEYE_H
