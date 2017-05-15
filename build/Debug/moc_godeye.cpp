/****************************************************************************
** Meta object code from reading C++ file 'godeye.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.2.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../godeye.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'godeye.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.2.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_GodEye_t {
    QByteArrayData data[8];
    char stringdata[64];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    offsetof(qt_meta_stringdata_GodEye_t, stringdata) + ofs \
        - idx * sizeof(QByteArrayData) \
    )
static const qt_meta_stringdata_GodEye_t qt_meta_stringdata_GodEye = {
    {
QT_MOC_LITERAL(0, 0, 6),
QT_MOC_LITERAL(1, 7, 7),
QT_MOC_LITERAL(2, 15, 0),
QT_MOC_LITERAL(3, 16, 10),
QT_MOC_LITERAL(4, 27, 13),
QT_MOC_LITERAL(5, 41, 7),
QT_MOC_LITERAL(6, 49, 9),
QT_MOC_LITERAL(7, 59, 3)
    },
    "GodEye\0openImg\0\0imgPredict\0updateStatues\0"
    "message\0showImage\0img\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_GodEye[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   34,    2, 0x08,
       3,    0,   35,    2, 0x08,
       4,    1,   36,    2, 0x0a,
       6,    1,   39,    2, 0x0a,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::QString,    5,
    QMetaType::Void, QMetaType::QImage,    7,

       0        // eod
};

void GodEye::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        GodEye *_t = static_cast<GodEye *>(_o);
        switch (_id) {
        case 0: _t->openImg(); break;
        case 1: _t->imgPredict(); break;
        case 2: _t->updateStatues((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: _t->showImage((*reinterpret_cast< const QImage(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject GodEye::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_GodEye.data,
      qt_meta_data_GodEye,  qt_static_metacall, 0, 0}
};


const QMetaObject *GodEye::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *GodEye::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GodEye.stringdata))
        return static_cast<void*>(const_cast< GodEye*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int GodEye::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 4)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
