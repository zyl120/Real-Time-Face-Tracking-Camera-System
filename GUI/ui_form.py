# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'form.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_GUI(object):
    def setupUi(self, GUI):
        GUI.setObjectName("GUI")
        GUI.resize(800, 600)
        self.verticalLayout = QtWidgets.QVBoxLayout(GUI)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(0, 100))
        self.label.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setPointSize(35)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.auto_button = QtWidgets.QRadioButton(GUI)
        self.auto_button.setMinimumSize(QtCore.QSize(0, 50))
        self.auto_button.setMaximumSize(QtCore.QSize(16777215, 50))
        self.auto_button.setObjectName("auto_button")
        self.horizontalLayout_3.addWidget(self.auto_button)
        self.manual_button = QtWidgets.QRadioButton(GUI)
        self.manual_button.setMinimumSize(QtCore.QSize(0, 50))
        self.manual_button.setMaximumSize(QtCore.QSize(16777215, 50))
        self.manual_button.setChecked(True)
        self.manual_button.setObjectName("manual_button")
        self.horizontalLayout_3.addWidget(self.manual_button)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.motorPosLabel = QtWidgets.QLabel(GUI)
        self.motorPosLabel.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.motorPosLabel.setObjectName("motorPosLabel")
        self.verticalLayout.addWidget(self.motorPosLabel)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_Left = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Left.sizePolicy().hasHeightForWidth())
        self.pushButton_Left.setSizePolicy(sizePolicy)
        self.pushButton_Left.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton_Left.setFont(font)
        self.pushButton_Left.setObjectName("pushButton_Left")
        self.horizontalLayout.addWidget(self.pushButton_Left)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.pushButton_Up = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Up.sizePolicy().hasHeightForWidth())
        self.pushButton_Up.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton_Up.setFont(font)
        self.pushButton_Up.setObjectName("pushButton_Up")
        self.verticalLayout_3.addWidget(self.pushButton_Up)
        self.pushButton_Center = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Center.sizePolicy().hasHeightForWidth())
        self.pushButton_Center.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton_Center.setFont(font)
        self.pushButton_Center.setObjectName("pushButton_Center")
        self.verticalLayout_3.addWidget(self.pushButton_Center)
        self.pushButton_Down = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Down.sizePolicy().hasHeightForWidth())
        self.pushButton_Down.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton_Down.setFont(font)
        self.pushButton_Down.setObjectName("pushButton_Down")
        self.verticalLayout_3.addWidget(self.pushButton_Down)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.pushButton_Right = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Right.sizePolicy().hasHeightForWidth())
        self.pushButton_Right.setSizePolicy(sizePolicy)
        self.pushButton_Right.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(25)
        self.pushButton_Right.setFont(font)
        self.pushButton_Right.setObjectName("pushButton_Right")
        self.horizontalLayout.addWidget(self.pushButton_Right)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.shutdownButton = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shutdownButton.sizePolicy().hasHeightForWidth())
        self.shutdownButton.setSizePolicy(sizePolicy)
        self.shutdownButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.shutdownButton.setFont(font)
        self.shutdownButton.setStyleSheet("color:red")
        self.shutdownButton.setObjectName("shutdownButton")
        self.horizontalLayout_2.addWidget(self.shutdownButton)
        self.rebootButton = QtWidgets.QPushButton(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rebootButton.sizePolicy().hasHeightForWidth())
        self.rebootButton.setSizePolicy(sizePolicy)
        self.rebootButton.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.rebootButton.setFont(font)
        self.rebootButton.setStyleSheet("color:yellow")
        self.rebootButton.setObjectName("rebootButton")
        self.horizontalLayout_2.addWidget(self.rebootButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(GUI)
        QtCore.QMetaObject.connectSlotsByName(GUI)

    def retranslateUi(self, GUI):
        _translate = QtCore.QCoreApplication.translate
        GUI.setWindowTitle(_translate("GUI", "GUI"))
        self.label.setText(_translate("GUI", "Camera Control"))
        self.auto_button.setText(_translate("GUI", "Auto"))
        self.manual_button.setText(_translate("GUI", "Manual"))
        self.motorPosLabel.setText(_translate("GUI", "Motor X: N/A, Y: N/A"))
        self.pushButton_Left.setText(_translate("GUI", "Left"))
        self.pushButton_Up.setText(_translate("GUI", "Up"))
        self.pushButton_Center.setText(_translate("GUI", "Center"))
        self.pushButton_Down.setText(_translate("GUI", "Down"))
        self.pushButton_Right.setText(_translate("GUI", "Right"))
        self.shutdownButton.setText(_translate("GUI", "Shutdown"))
        self.rebootButton.setText(_translate("GUI", "Reboot"))
