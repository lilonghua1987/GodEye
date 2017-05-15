#include "godeye.h"
#include <QApplication>
#include <QDesktopWidget>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    GodEye w;
    //w.adjustSize();
    w.move(QApplication::desktop()->screen()->rect().center() - w.rect().center());
    w.setFixedSize(w.size());
    w.show();

    return a.exec();
}
