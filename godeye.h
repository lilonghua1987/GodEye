#ifndef GODEYE_H
#define GODEYE_H

#include <QMainWindow>
#include "qrecogonition.h"

namespace Ui {
class GodEye;
}

class GodEye : public QMainWindow
{
    Q_OBJECT

public:
    explicit GodEye(QWidget *parent = 0);
    ~GodEye();

private:
    void setParams();

private slots:
    //void on_runButton_clicked();
    void openImg();
    void imgPredict();

public slots:
    void updateStatues(const QString& message);
    void showImage(const QImage& img);

private:
    Ui::GodEye *ui;
    QString fileName;
    //QRecogonition qreg;
    des::DESCIPTOR_METHOD fm;
    MATCH_METHOD mm;
    CLASSIFIER cm;
};

#endif // GODEYE_H
