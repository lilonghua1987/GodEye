#include "godeye.h"
#include "ui_godeye.h"
#include <QtGui>
#include <QMessageBox>
#include <QFileDialog>
#include <QEventLoop>

GodEye::GodEye(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GodEye)
{
    ui->setupUi(this);
    connect(ui->imgButton, SIGNAL(clicked()), this, SLOT(openImg()));
    connect(ui->runButton, SIGNAL(clicked()), this, SLOT(imgPredict()));
    //ui->menuBar->setNativeMenuBar(false);
}

GodEye::~GodEye()
{
    delete ui;
}


void GodEye::setParams()
{
    //FEATURE_METHOD fm = FEATURE_METHOD::SITF;
    //MATCH_METHOD mm = MATCH_METHOD::FlannBased;

    if (ui->sift->isChecked()){
        fm = des::SIFT;
    }else if (ui->surf->isChecked()) {
        fm = des::SURF;
    }else if (ui->brief->isChecked()) {
        fm = des::BRIEF;
    }else if (ui->brisk->isChecked()) {
        fm = des::BRISK;
    }else if (ui->freak->isChecked()) {
        fm = des::FREAK;
    }else if (ui->orb->isChecked()) {
        fm = des::ORB;
    }

    if (ui->flann->isChecked()){
        mm = MATCH_METHOD::FlannBased;
    }else if (ui->brute->isChecked()) {
        mm = MATCH_METHOD::BruteForce;
    }else if (ui->brute1->isChecked()) {
        mm = MATCH_METHOD::BruteForceL1;
    }else if (ui->bruteH1->isChecked()) {
        mm = MATCH_METHOD::BruteForceHamming1;
    }else if (ui->bruteH2->isChecked()) {
        mm = MATCH_METHOD::BruteForceHamming2;
    }

    if (ui->svm->isChecked()){
        cm = CLASSIFIER::SVM;
    }else if (ui->boost->isChecked()) {
        cm = CLASSIFIER::BOOST;
    }else if (ui->knn->isChecked()) {
        cm = CLASSIFIER::KNN;
    }else if (ui->ann->isChecked()) {
        cm = CLASSIFIER::ANN;
    }
}


void GodEye::imgPredict()
{
    if (!fileName.isEmpty()) {

        setParams();

        //qreg = QRecogonition("/home/lilonghua/softwares/GodEye/images/11");
        QRecogonition qreg("/home/lilonghua/softwares/GodEye/images/11",fm, mm, cm);
        qreg.setPredictImg(fileName.toStdString());
        connect(&qreg, SIGNAL(sendStatues(QString)), this, SLOT(updateStatues(QString)));
        connect(&qreg, SIGNAL(showStatues(QImage)), this, SLOT(showImage(QImage)));
        qreg.start();
        QEventLoop eventLoop;
        connect(&qreg,SIGNAL(finished ()),&eventLoop,SLOT(quit()));
        qreg.wait(1);
        eventLoop.exec();
    }
}

void GodEye::openImg()
{
    fileName = QFileDialog::getOpenFileName(this,tr("Open Image"), QDir::currentPath(),tr("Image Files (*.png *.jpg *.bmp)"));

    if (!fileName.isEmpty()){
        QImage image(fileName);
        if (image.isNull()) {
            QMessageBox::information(this, tr("Image Viewer"), tr("Cannot load %1.").arg(fileName));
            return;
        }

        if (ui->srcImgLabel->height() < image.height() || ui->srcImgLabel->width() < image.width())
        {
            ui->srcImgLabel->setPixmap(QPixmap::fromImage(image.scaled(ui->srcImgLabel->size())));
        }else{
            ui->srcImgLabel->setPixmap(QPixmap::fromImage(image.scaled(ui->srcImgLabel->size())));
        }
    }
}

void GodEye::updateStatues(const QString& message)
{
    this->ui->statusBar->showMessage(message);
}

void GodEye::showImage(const QImage &img)
{
    if (ui->resultImgLable->height() < img.height() || ui->resultImgLable->width() < img.width())
    {
        ui->resultImgLable->setPixmap(QPixmap::fromImage(img.scaled(ui->resultImgLable->size())));
    }else{
        this->ui->resultImgLable->setPixmap(QPixmap::fromImage(img));
    }
}
