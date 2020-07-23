import sys
import os
import time
import threading
import multiprocessing
import numpy as np
import PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *
from AircraftWar import game
from DQN_brain import DeepQNetwork


class MainWindow(QMainWindow):
    updatePrompt1Signal = pyqtSignal(str)
    updatePrompt2Signal = pyqtSignal(str)
    resetGUIParamSignal = pyqtSignal()
    xs = []
    ys = []
    updatePlotSignal = pyqtSignal()

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.initWindow(self)
        self.initModules()
        return

    def initModules(self):
        return

    def __del__(self):
        return

    def initWindow(self, Form):
        # 主窗口
        Form.setObjectName("MainWindow")
        Form.resize(1000, 500)
        Form.setMinimumSize(QtCore.QSize(1000, 500))
        Form.setMaximumSize(QtCore.QSize(1000, 500))
        Form.setWindowTitle("DQN AirCraft War")
        Form.setWindowIcon(QIcon('./resources/TensorFlow.png'))

        # 信息显示框
        self.label_prompt = QtWidgets.QLabel(Form)
        self.label_prompt.setGeometry(QtCore.QRect(10, 0, 300, 30))
        self.label_prompt.setTextFormat(QtCore.Qt.AutoText)
        self.label_prompt.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_prompt.setObjectName("label_prompt")
        self.label_prompt.setText("Prompt")

        self.textEdit_prompt1 = QtWidgets.QTextEdit(Form)
        self.textEdit_prompt1.setGeometry(QtCore.QRect(10, 30, 300, 225))
        self.textEdit_prompt1.setObjectName("Prompt1 View Box")
        self.textEdit_prompt1.setReadOnly(True)
        self.updatePrompt1Signal.connect(self.updatePrompt1Func)
        self.textEdit_prompt2 = QtWidgets.QTextEdit(Form)
        self.textEdit_prompt2.setGeometry(QtCore.QRect(10, 265, 300, 225))
        self.textEdit_prompt2.setObjectName("Prompt2 View Box")
        self.textEdit_prompt2.setReadOnly(True)
        self.updatePrompt2Signal.connect(self.updatePrompt2Func)

        self.plotchart_scores = PyQt5.QtChart.QChartView(Form)
        self.plotchart_scores.setGeometry(QtCore.QRect(310, 0, 380, 250))
        self.plotchart_scores.setObjectName("plotchart_scores")
        self.plotchart_losses = PyQt5.QtChart.QChartView(Form)
        self.plotchart_losses.setGeometry(QtCore.QRect(310, 250, 380, 250))
        self.plotchart_losses.setObjectName("plotchart_losses")
        self.updatePlotSignal.connect(self.updatePlot)

        # 可调的参数
        self.label_observation = QtWidgets.QLabel(Form)
        self.label_observation.setGeometry(QtCore.QRect(690, 0, 300, 30))
        self.label_observation.setTextFormat(QtCore.Qt.AutoText)
        self.label_observation.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_observation.setObjectName("observation")
        self.label_observation.setText("Observation")
        self.comboBox_observation = QtWidgets.QComboBox(Form)
        self.comboBox_observation.setGeometry(QtCore.QRect(690, 30, 300, 30))
        self.comboBox_observation.setObjectName("Observation")
        self.comboBox_observation.addItem("All Planes")
        self.comboBox_observation.setEditable(False)

        self.label_reward_decay = QtWidgets.QLabel(Form)
        self.label_reward_decay.setGeometry(QtCore.QRect(690, 60, 300, 30))
        self.label_reward_decay.setTextFormat(QtCore.Qt.AutoText)
        self.label_reward_decay.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_reward_decay.setObjectName("reward_decay")
        self.label_reward_decay.setText("Reward Decay")
        self.spinbox_reward_decay = QtWidgets.QDoubleSpinBox(Form)
        self.spinbox_reward_decay.setGeometry(QtCore.QRect(690, 90, 300, 30))
        self.spinbox_reward_decay.setRange(0.0, 1.0)
        self.spinbox_reward_decay.setSingleStep(0.001)
        self.spinbox_reward_decay.setValue(0.9)
        self.spinbox_reward_decay.setObjectName("reward_decay")
        self.spinbox_reward_decay.setDecimals(3)

        self.label_e_greedy = QtWidgets.QLabel(Form)
        self.label_e_greedy.setGeometry(QtCore.QRect(690, 120, 300, 30))
        self.label_e_greedy.setTextFormat(QtCore.Qt.AutoText)
        self.label_e_greedy.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_e_greedy.setObjectName("e_greedy")
        self.label_e_greedy.setText("Epsilon Greedy")
        self.spinbox_e_greedy = QtWidgets.QDoubleSpinBox(Form)
        self.spinbox_e_greedy.setGeometry(QtCore.QRect(690, 150, 300, 30))
        self.spinbox_e_greedy.setRange(0.0, 1.0)
        self.spinbox_e_greedy.setSingleStep(0.001)
        self.spinbox_e_greedy.setValue(0.9)
        self.spinbox_e_greedy.setObjectName("e_greedy")
        self.spinbox_e_greedy.setDecimals(3)

        self.label_replace_target_iter = QtWidgets.QLabel(Form)
        self.label_replace_target_iter.setGeometry(
            QtCore.QRect(690, 180, 300, 30))
        self.label_replace_target_iter.setTextFormat(QtCore.Qt.AutoText)
        self.label_replace_target_iter.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_replace_target_iter.setObjectName("replace_target_iter")
        self.label_replace_target_iter.setText("Replace Target Iterations")
        self.spinbox_replace_target_iter = QtWidgets.QSpinBox(Form)
        self.spinbox_replace_target_iter.setGeometry(
            QtCore.QRect(690, 210, 300, 30))
        self.spinbox_replace_target_iter.setRange(0, 100000)
        self.spinbox_replace_target_iter.setSingleStep(1)
        self.spinbox_replace_target_iter.setValue(1000)
        self.spinbox_replace_target_iter.setObjectName("replace_target_iter")

        self.label_memory_size = QtWidgets.QLabel(Form)
        self.label_memory_size.setGeometry(QtCore.QRect(690, 240, 300, 30))
        self.label_memory_size.setTextFormat(QtCore.Qt.AutoText)
        self.label_memory_size.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_memory_size.setObjectName("memory_size")
        self.label_memory_size.setText("Memory Size")
        self.spinbox_memory_size = QtWidgets.QSpinBox(Form)
        self.spinbox_memory_size.setGeometry(QtCore.QRect(690, 270, 300, 30))
        self.spinbox_memory_size.setRange(0, 100000)
        self.spinbox_memory_size.setSingleStep(1)
        self.spinbox_memory_size.setValue(1500)
        self.spinbox_memory_size.setObjectName("memory_size")

        self.label_reward_set = QtWidgets.QLabel(Form)
        self.label_reward_set.setGeometry(QtCore.QRect(690, 300, 300, 30))
        self.label_reward_set.setTextFormat(QtCore.Qt.AutoText)
        self.label_reward_set.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_reward_set.setObjectName("reward_set")
        self.label_reward_set.setText("Reward Settings")

        self.label_score_reward = QtWidgets.QLabel(Form)
        self.label_score_reward.setGeometry(QtCore.QRect(690, 330, 150, 30))
        self.label_score_reward.setTextFormat(QtCore.Qt.AutoText)
        self.label_score_reward.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_score_reward.setObjectName("score_reward")
        self.label_score_reward.setText("Score Reward Ratio")
        self.spinbox_score_reward = QtWidgets.QDoubleSpinBox(Form)
        self.spinbox_score_reward.setObjectName("score_reward")
        self.spinbox_score_reward.setGeometry(QtCore.QRect(850, 330, 140, 30))
        self.spinbox_score_reward.setDecimals(4)
        self.spinbox_score_reward.setRange(-10000000.0, 10000000.0)
        self.spinbox_score_reward.setSingleStep(0.001)
        self.spinbox_score_reward.setValue(0.001)

        self.label_death_reward = QtWidgets.QLabel(Form)
        self.label_death_reward.setGeometry(QtCore.QRect(690, 360, 150, 30))
        self.label_death_reward.setTextFormat(QtCore.Qt.AutoText)
        self.label_death_reward.setAlignment(
            QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.label_death_reward.setObjectName("death_reward")
        self.label_death_reward.setText("Death Reward")
        self.spinbox_death_reward = QtWidgets.QDoubleSpinBox(Form)
        self.spinbox_death_reward.setObjectName("death_reward")
        self.spinbox_death_reward.setGeometry(QtCore.QRect(850, 360, 140, 30))
        self.spinbox_death_reward.setDecimals(3)
        self.spinbox_death_reward.setRange(-10000000.0, 10000000.0)
        self.spinbox_death_reward.setSingleStep(0.001)
        self.spinbox_death_reward.setValue(-10000.0)

        # 按钮
        self.button_set = QtWidgets.QPushButton(Form)
        self.button_set.setGeometry(QtCore.QRect(690, 410, 150, 80))
        self.button_set.setObjectName("button_set")
        self.button_set.setText("Set")
        self.button_set.clicked.connect(self.func_set)
        self.button_reset = QtWidgets.QPushButton(Form)
        self.button_reset.setGeometry(QtCore.QRect(850, 410, 50, 80))
        self.button_reset.setObjectName("button_reset")
        self.button_reset.setText("Reset")
        self.button_reset.clicked.connect(self.func_reset)
        self.button_plot = QtWidgets.QPushButton(Form)
        self.button_plot.setGeometry(QtCore.QRect(910, 410, 80, 80))
        self.button_plot.setObjectName("button_plot")
        self.button_plot.setText("Plot\nLoss")
        self.button_plot.clicked.connect(self.func_plot)

        self.resetGUIParamSignal.connect(self.resetGUIParam)

        self.show()
        return

    def updatePrompt1Func(self, str):
        self.textEdit_prompt1.append(str)
        return

    def updatePrompt2Func(self, str):
        self.textEdit_prompt2.append(str)
        return

    def prompt_print(self, content, color_timestamp='#00BFFF', color_content='#696969', promptID=1):
        timestamp = "<font color=\"%s\">" % (color_timestamp) + \
            time.strftime("%x %X:") + "</font>"
        content = "<font color=\"%s\">" % (color_content) + \
            content + "</font>"
        if promptID == 1:
            self.updatePrompt1Signal.emit(timestamp)
            self.updatePrompt1Signal.emit(content)
        elif promptID == 2:
            self.updatePrompt2Signal.emit(timestamp)
            self.updatePrompt2Signal.emit(content)
        return

    def func_set(self):
        r_d = float(self.spinbox_reward_decay.value())
        e_g = float(self.spinbox_e_greedy.value())
        r_t_i = int(self.spinbox_replace_target_iter.value())
        m_s = int(self.spinbox_memory_size.value())
        num0 = float(self.spinbox_score_reward.value())
        num1 = float(self.spinbox_death_reward.value())
        self.prompt_print(
            "Parameters set to:\r\n<br/>" +
            "Reward Decay: %f\r\n<br/>" % (r_d) +
            "Epsilon Greedy: %f\r\n<br/>" % (e_g) +
            "Replace Target Iters: %d\r\n<br/>" % (r_t_i) +
            "Memory Size: %d\r\n<br/>" % (m_s) +
            "Score Reward: %f\r\n<br/>" % (num0) +
            "Death Reward: %f" % (num1))
        while game_env.stepping:
            # 等待游戏画面刷新完成
            continue
        DQN.set_params(r_d, e_g, r_t_i, m_s)
        game_env.set_reward(num0, num1)
        return

    def resetGUIParam(self):
        self.spinbox_reward_decay.setValue(0.9)
        self.spinbox_e_greedy.setValue(0.9)
        self.spinbox_memory_size.setValue(1000)
        self.spinbox_replace_target_iter.setValue(1500)
        self.spinbox_score_reward.setValue(0.001)
        self.spinbox_death_reward.setValue(-10000.0)
        return

    def func_reset(self):
        self.resetGUIParamSignal.emit()

        while game_env.stepping:
            # 等待游戏画面刷新完成
            continue

        DQN.set_params(0.9, 0.9, 1000, 1500)
        game_env.set_reward(0.001, -10000.0)
        self.prompt_print("Parameters reset.")
        return

    def func_plot(self):
        DQN.plot_cost()  # 显示图表
        self.prompt_print("Figures ploted.")
        return

    def updatePlot(self):
        # plot scores
        self.series_scores = PyQt5.QtChart.QLineSeries()
        self.series_scores.setName("Scores")

        datanum = 100
        if len(self.xs) < datanum:
            xmin, xmax = 0, len(self.xs)-1
            ymin, ymax = min(self.ys[xmin: xmax]), max(self.ys[xmin: xmax])
            for i in range(len(self.xs)):
                self.series_scores.append(self.xs[i], self.ys[i])
        else:
            xmin, xmax = len(self.xs)-datanum+1, len(self.xs)
            ymin, ymax = min(self.ys[xmin: xmax]), max(self.ys[xmin: xmax])
            for i in range(len(self.xs)-datanum+1, len(self.xs)):
                self.series_scores.append(self.xs[i], self.ys[i])

        self.x_Aix_scores = PyQt5.QtChart.QValueAxis()
        self.x_Aix_scores.setRange(xmin, xmax)
        self.x_Aix_scores.setLabelFormat("%d")
        self.y_Aix_scores = PyQt5.QtChart.QValueAxis()
        self.y_Aix_scores.setRange(ymin, ymax)
        self.y_Aix_scores.setLabelFormat("%d")

        self.plotchart_scores.chart().removeAllSeries()
        self.plotchart_scores.chart().addSeries(self.series_scores)
        self.plotchart_scores.chart().setAxisX(self.x_Aix_scores)
        self.plotchart_scores.chart().setAxisY(self.y_Aix_scores)
        self.plotchart_scores.show()

        # plot losses
        self.series_losses = PyQt5.QtChart.QLineSeries()
        self.series_losses.setName("Loss")

        losses = DQN.get_cost()
        xx = np.arange(len(losses))
        datanum = 1000
        if len(xx) < datanum:
            xmin, xmax = 0, len(xx)-1
            ymin, ymax = min(losses[xmin: xmax]), max(losses[xmin: xmax])
            for i in range(len(xx)):
                self.series_losses.append(xx[i], losses[i])
        else:
            xmin, xmax = len(xx)-datanum+1, len(xx)
            ymin, ymax = min(losses[xmin: xmax]), max(losses[xmin: xmax])
            for i in range(len(xx)-datanum+1, len(xx)):
                self.series_losses.append(xx[i], losses[i])

        self.x_Aix_losses = PyQt5.QtChart.QValueAxis()
        self.x_Aix_losses.setRange(xmin, xmax)
        self.x_Aix_losses.setLabelFormat("%d")
        self.y_Aix_losses = PyQt5.QtChart.QValueAxis()
        self.y_Aix_losses.setRange(ymin, ymax)
        self.y_Aix_losses.setLabelFormat("%d")

        self.plotchart_losses.chart().removeAllSeries()
        self.plotchart_losses.chart().addSeries(self.series_losses)
        self.plotchart_losses.chart().setAxisX(self.x_Aix_losses)
        self.plotchart_losses.chart().setAxisY(self.y_Aix_losses)
        self.plotchart_losses.show()

        return

    def updatePlotData(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        if len(self.xs) > 1:
            self.updatePlotSignal.emit()
        return


# 初始化游戏
game_env = game()
# 初始化DQN网络
DQN = DeepQNetwork(game_env.n_actions, game_env.n_features,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.8,
                  replace_target_iter=1000,
                  memory_size=1500,
                  output_graph=True
                  )

save_checkpoint = False  # 设定是否保存记录


def run_DQN():

    # 读取checkpoint
    # DQN.load_model('./saved_models/model-54000pts-2020-06-11-15-10-16.ckpt')

    fig_x = []
    fig_y = []

    highestScore = 0

    step = 0

    for episode in range(30000):
        if episode < 25000:
            game_env.set_tick(600)
        else:
            game_env.set_tick(60)

        # initial observation
        observation = np.zeros([20])
        # observation = np.array([0., 0.])

        while True:
            # DQN choose action based on observation
            action = DQN.choose_action(observation)

            # DQN take action and get next observation and reward
            observation_, reward, done = game_env.step(action)
            time.sleep(0.01)

            DQN.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                DQN.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                thisScore = game_env.restart()
                print("Episode %5d: %d Scores" % (episode, thisScore))
                mainWindow.prompt_print("Episode %5d: %d Scores" % (
                    episode, thisScore), promptID=2)
                mainWindow.updatePlotData(episode, thisScore)

                if save_checkpoint:  # 设定是否保存记录
                    if thisScore > highestScore:
                        highestScore = thisScore
                        strTime = time.strftime(
                            "%Y-%m-%d-%H-%M-%S", time.localtime())
                        filepath = './saved_models/model-' + \
                            str(highestScore) + 'pts-' + strTime + '.ckpt'
                        DQN.save_model(filepath)

                break
            step += 1

    # end of game
    print('game over')
    # pygame.quit()
    DQN.plot_cost()


if __name__ == '__main__':
    t1 = threading.Thread(target=run_DQN)  # 开个线程运行游戏
    t1.setDaemon(True)
    t1.start()
    app = QApplication(sys.argv)  # 打开GUI
    mainWindow = MainWindow(app)
    app.exec_()
