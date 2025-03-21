import numpy as np
import pandas as pd
import numpy as np
import time
import sys
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 充电和休息奖励高低



class Maze():

    def reset(self):
        # 无人机状态
        #
        # S = 0，1，2 /0：通信，1：充电 2：休息
        self.ori_reward = 0
        Battery = 600
        S0 = 1
        T0 = 4
        self.rect0 = [Battery, S0, T0]
        self.rect1 = [Battery, S0, T0]
        self.rect2 = [Battery, S0, T0]
        self.rect3 = [Battery, S0, T0]
        self.rect4 = [Battery, S0, T0]
        self.rect5 = [Battery, S0, T0]
        self.rect6 = [Battery, S0, T0]
        self.rect7 = [Battery, S0, T0]
        self.rect8 = [Battery, S0, T0]
        self.rect9 = [Battery, S0, T0]
        self.rect10 = [Battery, S0, T0]
        self.rect11 = [Battery, S0, T0]
        self.rect12 = [Battery, S0, T0]
        self.rect13 = [Battery, S0, T0]
        self.rect14 = [Battery, S0, T0]
        # self.rect8 = ([Battery, S0, T0])
        # input 012
        # 5个input 维度
        self.count = 0

        return self.rect0 + self.rect1 + self.rect2 + self.rect3 + self.rect4 + self.rect5 + self.rect6 + self.rect7 + self.rect8 + self.rect9 + self.rect10 + self.rect11 + self.rect12 + self.rect13 + self.rect14

    def getstate(self):
        self.rectm0 = [self.rect0[0]] + [self.rect0[1]]
        self.rectm1 = [self.rect1[0]] + [self.rect1[1]]
        self.rectm2 = [self.rect2[0]] + [self.rect2[1]]
        self.rectm3 = [self.rect3[0]] + [self.rect3[1]]
        self.rectm4 = [self.rect4[0]] + [self.rect4[1]]
        self.rectm5 = [self.rect5[0]] + [self.rect5[1]]
        self.rectm6 = [self.rect6[0]] + [self.rect6[1]]
        self.rectm7 = [self.rect7[0]] + [self.rect7[1]]
        self.rectm8 = [self.rect8[0]] + [self.rect8[1]]
        self.rectm9 = [self.rect9[0]] + [self.rect9[1]]
        self.rectm10 = [self.rect10[0]] + [self.rect10[1]]
        self.rectm11 = [self.rect11[0]] + [self.rect11[1]]
        self.rectm12 = [self.rect12[0]] + [self.rect12[1]]
        self.rectm13 = [self.rect13[0]] + [self.rect13[1]]
        self.rectm14 = [self.rect14[0]] + [self.rect14[1]]
        self.T = [self.rect0[2]]
        return self.rectm0 + self.rectm1 + self.rectm2 + self.rectm3 + self.rectm4 + self.rectm5 + self.rectm6 + self.rectm7 + self.rectm8 + self.rectm9 + self.rectm10 + self.rectm11 + self.rectm12 + self.rectm13 + self.rectm14 + self.T

    def step(self, action):
        # 接收三个动作的
        # action = [0,1,3]
        self.count += 1

        s0 = self.rect0
        s1 = self.rect1
        s2 = self.rect2
        s3 = self.rect3
        s4 = self.rect4
        s5 = self.rect5
        s6 = self.rect6
        s7 = self.rect7
        s8 = self.rect8
        s9 = self.rect9
        s10 = self.rect10
        s11 = self.rect11
        s12 = self.rect12
        s13 = self.rect13
        s14 = self.rect14
        # s2 = self.canvas.coords(self.rect2)
        s_lis = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14]
        # s_all = [s0,s1,s2]
        base_action = np.array([[0, 0, 0] for i in range(15)])
        s = action

        # input normalization调整输入维度

        for i in range(15):
            # t = s % 3 # S = 0，1，2 /
            # 0：通信. 1：充电/休息
            ac = action[i]

            if 0 >= ac >= -1:
                UAVac = 1
            if 1 >= ac > 0:
                UAVac = 0

            # if s_lis[i][0] <= 64:
            #     UAVac = 1

            if UAVac == 0:  # 动作为通信
                if s_lis[i][1] == 0:  # 当前状态为通信
                    if s_lis[i][0] - 211 >= 30:  # 如果大于，即正常作动作，如果小于，则休息或充电
                        base_action[i][0] -= 211  # 电量B
                        base_action[i][1] == base_action[i][1]  # 状态S
                        base_action[i][2] += 1  # 时间T
                    if s_lis[i][0] - 211 < 30:  # 充电
                        # if s_lis[i][2] == 7:  # 时刻为白天，动作为充电
                            # base_action[i][0] -= 82  # 电量153*11/12-402*1/12-206*11/12
                            # base_action[i][1] += 1  # 状态
                            # base_action[i][2] += 1  # 时间T
                        if s_lis[i][2] == 8:
                            if s_lis[i][0] + 31 < 600:
                                base_action[i][0] += 31  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 31 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 9:

                            if s_lis[i][0] + 121 < 600:
                                base_action[i][0] += 121  # 电量375*(3600-275)/3600-402*275/3600-211*(3600-275)/3600
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 121 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 10:

                            if s_lis[i][0] + 185 < 600:
                                base_action[i][0] += 185  # 电量444*11/12-222
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 185 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 11:

                            if s_lis[i][0] + 224 < 600:
                                base_action[i][0] += 224  # 电量486*11/12-402-206*11/12
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 224 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 12:

                            if s_lis[i][0] + 236 < 600:
                                base_action[i][0] += 224  # 电量500*11/12-402-206*11/12
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 236 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 13:
                            if s_lis[i][0] + 224 < 600:
                                base_action[i][0] += 224  # 电量486*11/12-402-206*11/12
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 224 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 14:
                            if s_lis[i][0] + 185 < 600:
                                base_action[i][0] += 185  # 电量444*11/12-222
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 185 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 15:
                            if s_lis[i][0] + 121 < 600:
                                base_action[i][0] += 121  # 电量375*11/12-402/12-206*11/12
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 121 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 16:
                            if s_lis[i][0] + 31 < 600:
                                base_action[i][0] += 31  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 31 > 600:
                                base_action[i][0] += 600 - s_lis[i][
                                    0]  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                                base_action[i][1] += 1  # 状态
                                base_action[i][2] += 1  # 时间
                        # elif s_lis[i][2] == 17:
                        #     base_action[i][0] -= 82  #
                        #     base_action[i][1] += 1
                        #     base_action[i][2] += 1
                        else:
                            base_action[i][0] == base_action[i][0]  # 夜间休息状态
                            base_action[i][1] += 1
                            base_action[i][2] += 1

                if s_lis[i][1] == 1:  # 当前状态为休息或充电
                    if s_lis[i][0] - 215 >= 30:  # 如果大于，则正常通信
                        if 0 <= s_lis[i][2] <= 7 or 17 <= s_lis[i][2] <= 23:  # 当前状态为夜间休息：从休息到通信
                            base_action[i][0] -= 215  # 电量B ：0.7745
                            base_action[i][1] -= 1  # 状态S
                            base_action[i][2] += 1  # 时间T
                        elif 7 < s_lis[i][2] < 17:  # 当前状态为白天充电：从充电到通信
                            base_action[i][0] -= 196  # 电量B ：0.7059
                            base_action[i][1] -= 1  # 状态S
                            base_action[i][2] += 1  # 时间T
                    if s_lis[i][0] - 215 < 30:  # 如果小于，则休息或者充电
                        # if s_lis[i][2] == 7:
                        #     base_action[i][0] -= 39  # 电量153-206
                        #     base_action[i][1] == base_action[i][1]  # 状态
                        #     base_action[i][2] += 1  # 时间T
                        if s_lis[i][2] == 8:

                            if s_lis[i][0] + 21 < 600:
                                base_action[i][0] += 21  # 电量278-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 21 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 9:
                            if s_lis[i][0] + 164 < 600:
                                base_action[i][0] += 164  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 164 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 10:

                            if s_lis[i][0] + 233 < 600:
                                base_action[i][0] += 233  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 233 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 11:

                            if s_lis[i][0] + 275 < 600:
                                base_action[i][0] += 275  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 275 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 12:

                            if s_lis[i][0] + 289 < 600:
                                base_action[i][0] += 289  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 289 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 13:
                            if s_lis[i][0] + 275 < 600:
                                base_action[i][0] += 275  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 275 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 14:
                            if s_lis[i][0] + 233 < 600:
                                base_action[i][0] += 233  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 233 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 15:
                            if s_lis[i][0] + 164 < 600:
                                base_action[i][0] += 164  # 电量375-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 164 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        elif s_lis[i][2] == 16:
                            if s_lis[i][0] + 67 < 600:
                                base_action[i][0] += 67  # 电量278-211
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                            if s_lis[i][0] + 67 > 600:
                                base_action[i][0] += 600 - s_lis[i][0]
                                base_action[i][1] == base_action[i][1]  # 状态
                                base_action[i][2] += 1  # 时间
                        # elif s_lis[i][2] == 17:
                        #     base_action[i][0] -= 53  #
                        #     base_action[i][1] == base_action[i][1]
                        #     base_action[i][2] += 1
                        # if s_lis[i][2] = 18，19，20，21，22，23，0，1，2，3，4，5，6
                        else:
                            base_action[i][0] == base_action[i][0]  #
                            base_action[i][1] == base_action[i][1]
                            base_action[i][2] += 1
                # elif s_lis[i][1] == 1: # 当前状态为充电
                #     base_action[i][0] -= 350
                #     base_action[i][1] -= 1
                #     base_action[i][2] += 1
                # elif s_lis[i][1] == 2:  # 当前状态为休息
                #     base_action[i][0] -= 350  # 设置步长为10
                #     base_action[i][1] -= 2
                #     base_action[i][2] += 1
            if UAVac == 1:  # 充电或休息
                if s_lis[i][1] == 0:  # 当前状态为通信
                    # if s_lis[i][2] == 7:  # 时刻为白天，动作为充电
                    #     base_action[i][0] -= 82  # 电量153*11/12-402*1/12-206*11/12
                    #     base_action[i][1] += 1  # 状态
                    #     base_action[i][2] += 1  # 时间T

                    # 改为上升到1400米，275s/3600s, 1*5静态功率，（休息）
                    # 0.0764,206改为211W
                    if s_lis[i][2] == 8:
                        if s_lis[i][0] + 31 < 600:
                            base_action[i][0] += 31  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 31 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 9:

                        if s_lis[i][0] + 121 < 600:
                            base_action[i][0] += 121  # 电量375*11/12-402/12-206*11/12
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 121 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 10:

                        if s_lis[i][0] + 185 < 600:
                            base_action[i][0] += 185  # 电量444*11/12-222
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 185 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 11:

                        if s_lis[i][0] + 224 < 600:
                            base_action[i][0] += 224  # 电量486*11/12-402-206*11/12
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 224 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 12:

                        if s_lis[i][0] + 236 < 600:
                            base_action[i][0] += 224  # 电量500*11/12-402-206*11/12
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 236 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 13:
                        if s_lis[i][0] + 224 < 600:
                            base_action[i][0] += 224  # 电量486*11/12-402-206*11/12
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 224 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 14:
                        if s_lis[i][0] + 185 < 600:
                            base_action[i][0] += 185  # 电量444*11/12-222
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 185 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 15:
                        if s_lis[i][0] + 121 < 600:
                            base_action[i][0] += 121  # 电量375*11/12-402/12-206*11/12
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 121 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 16:
                        if s_lis[i][0] + 31 < 600:
                            base_action[i][0] += 31  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 31 > 600:
                            base_action[i][0] += 600 - s_lis[i][
                                0]  # 电量278*11/12-402/12-206*11/12     ###（减8?)从休息到充电
                            base_action[i][1] += 1  # 状态
                            base_action[i][2] += 1  # 时间
                    else:
                        base_action[i][0] == base_action[i][0]  # 夜间休息状态
                        base_action[i][1] += 1
                        base_action[i][2] += 1

                if s_lis[i][1] == 1:  # 当前状态为充电或休息
                    # if s_lis[i][2] == 7:
                    #     base_action[i][0] -= 39  # 电量153-206
                    #     base_action[i][1] == base_action[i][1]  # 状态
                    #     base_action[i][2] += 1  # 时间T
                    if s_lis[i][2] == 8:

                        if s_lis[i][0] + 21 < 600:
                            base_action[i][0] += 21  # 电量278-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 21 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 9:
                        if s_lis[i][0] + 164 < 600:
                            base_action[i][0] += 164  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 164 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 10:
                        if s_lis[i][0] + 233 < 600:
                            base_action[i][0] += 233  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 233 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 11:

                        if s_lis[i][0] + 275 < 600:
                            base_action[i][0] += 275  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 275 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 12:

                        if s_lis[i][0] + 289 < 600:
                            base_action[i][0] += 289  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 289 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 13:
                        if s_lis[i][0] + 275 < 600:
                            base_action[i][0] += 275  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 275 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 14:
                        if s_lis[i][0] + 233 < 600:
                            base_action[i][0] += 233  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 233 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 15:
                        if s_lis[i][0] + 164 < 600:
                            base_action[i][0] += 164  # 电量375-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 164 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    elif s_lis[i][2] == 16:
                        if s_lis[i][0] + 67 < 600:
                            base_action[i][0] += 67  # 电量278-211
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                        if s_lis[i][0] + 67 > 600:
                            base_action[i][0] += 600 - s_lis[i][0]
                            base_action[i][1] == base_action[i][1]  # 状态
                            base_action[i][2] += 1  # 时间
                    # elif s_lis[i][2] == 17:
                    #     base_action[i][0] -= 1  #
                    #     base_action[i][1] == base_action[i][1]
                    #     base_action[i][2] += 1
                    else:
                        base_action[i][0] == base_action[i][1]  #
                        base_action[i][1] == base_action[i][1]
                        base_action[i][2] += 1
                # elif s_lis[i][1] == 2:  # 上一状态为休息
                #     if s_lis[i][2] == 7:
                #         base_action[i][0] -= 72  # 电量12.4*11/12-13.852
                #         base_action[i][1] -= 1  # 状态
                #         base_action[i][2] += 1  # 时间T
                #     elif s_lis[i][2] == 8:
                #         base_action[i][0] += 160  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1  #
                #     elif s_lis[i][2] == 9:
                #         base_action[i][0] += 233  # 电量11/12-13.852
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 10:
                #         base_action[i][0] += 284  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 11:
                #         base_action[i][0] += 313  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 12:
                #         base_action[i][0] += 324  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 13:
                #         base_action[i][0] += 313  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 14:
                #         base_action[i][0] += 284  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 15:
                #         base_action[i][0] += 233  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 16:
                #         base_action[i][0] += 160  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     elif s_lis[i][2] == 17:
                #         base_action[i][0] += 72  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1
                #     # if s_lis[i][2] = 18，19，20，21，22，23，0，1，2，3，4，5，6
                #
                #     else:
                #         base_action[i][0] -= 362  #
                #         base_action[i][1] -= 1
                #         base_action[i][2] += 1

            # elif UAVac == 2:  # 休息
            #     if s_lis[i][1] == 0: #上一状态为通信
            #         base_action[i][0] == base_action[i][0]  #电量B
            #         base_action[i][1] += 2  #状态S
            #         base_action[i][2] += 1  #时间T
            #     elif s_lis[i][1] == 1: #上一状态为充电
            #         base_action[i][0] == base_action[i][0] # 设置步长为10
            #         base_action[i][1] += 1
            #         base_action[i][2] += 1
            #     elif s_lis[i][1] == 2: #上一状态为休息
            #         base_action[i][0] == base_action[i][0]  # 设置步长为10
            #         base_action[i][1] == base_action[i][1]
            #         base_action[i][2] += 1

            # s = int(s//3)
            # for i in range(8)
            # s0 = self.rect0
            # s1 = self.rect1
            # s2 = self.rect2
            # s3 = self.rect3
            # s4 = self.rect4
            # s5 = self.rect5
            # s6 = self.rect6
            # s7 = self.rect7
            if s_lis[i][0] <= 30:
                base_action[i][0] = 0

                #base_action[i][1] =

        self.rect0 = self.rect0 + base_action[0]
        self.rect1 = self.rect1 + base_action[1]
        self.rect2 = self.rect2 + base_action[2]
        self.rect3 = self.rect3 + base_action[3]
        self.rect4 = self.rect4 + base_action[4]
        self.rect5 = self.rect5 + base_action[5]
        self.rect6 = self.rect6 + base_action[6]
        self.rect7 = self.rect7 + base_action[7]
        self.rect8 = self.rect8 + base_action[8]
        self.rect9 = self.rect9 + base_action[9]
        self.rect10 = self.rect10 + base_action[10]
        self.rect11 = self.rect11 + base_action[11]
        self.rect12 = self.rect12 + base_action[12]
        self.rect13 = self.rect13 + base_action[13]
        self.rect14 = self.rect14 + base_action[14]

        s_14 = list(self.rect14)
        s_13 = list(self.rect13)
        s_12 = list(self.rect12)
        s_11 = list(self.rect11)
        s_10 = list(self.rect10)
        s_9 = list(self.rect9)
        s_8 = list(self.rect8)
        s_7 = list(self.rect7)
        s_6 = list(self.rect6)
        s_5 = list(self.rect5)
        s_4 = list(self.rect4)
        s_3 = list(self.rect3)
        s_2 = list(self.rect2)
        s_1 = list(self.rect1)
        s_0 = list(self.rect0)

        # s_19 = s_9[0]+s_9[1]
        # s_18 = s_8[0]+s_8[1]
        # s_17 = s_7[0]+s_7[1]
        # s_16 = s_6[0]+s_6[1]
        # s_15 = s_5[0]+s_5[1]
        # s_14 = s_4[0]+s_4[1]
        # s_13 = s_3[0]+s_3[1]
        # s_12 = s_2[0]+s_2[1]
        # s_11 = s_1[0]+s_1[1]
        # s_10 = s_0[0]+s_0[1]
        # s_Tm = s_0[2]

        reward_sigle = 0
        dic = {}
        tx_count = 0
        UAVcount = 0
        relax_count = 0
        reward_cover = 0
        reward_solar = 0
        reward_relax = 0
        reward_punish = 0
        UAVcount1 = 0
        s_Location = [s_0[1], s_1[1], s_2[1], s_3[1], s_4[1], s_5[1], s_6[1], s_7[1], s_8[1], s_9[1], s_10[1], s_11[1],
                      s_12[1], s_13[1], s_14[1]]
        s_Battery = [s_0[0], s_1[0], s_2[0], s_3[0], s_4[0], s_5[0], s_6[0], s_7[0], s_8[0], s_9[0], s_10[0], s_11[0],
                     s_12[0], s_13[0], s_14[0]]
        # for o in range(15):
        #     if s_Battery[o] <= 30:
        #         s_Location[o] = 1
        for i in range(15):
            ac = action[i]
            if 1 >= ac > 0 and s_lis[i][0] - 215 <= 30:
                UAVcount += 1
            # if 0 >= ac > -1 and base_action[i][0] == 650 - s_lis[i][0]:
            #     UAVcount1 += 1

            # if s_Battery[m] <= 0:
            #     UAVcount += 1
        s_T = s_0[2]
        # a,b,c,d,e,f,g

        # 10，11，12，13，14 充电给正reward
        # 其他时间休息给正reward

        if UAVcount >= 3:
            reward_punish -= 100
        # if UAVcount1 >= 1:
        #     reward_sigle -= 100

        # 5, 21时段
        if s_T == 5 or s_T == 21:
            for a in range(15):

                if s_Location[a] == 0:
                    tx_count += 1
                    if s_Battery[a] <= 0:
                        tx_count -= 1

                if s_Location[a] == 1:
                    relax_count += 1
            # 不满足最低覆盖率为85%
            # 权衡硬约束和软约束？
            if tx_count == 0:
                reward_cover -= 30
            if tx_count == 1:
                reward_cover += 9

            if tx_count >= 2:
                reward_cover += 10

            # if relax_count <  10: # 10个无人机休息，给高奖励
            #     reward_relax -=20

            if relax_count >= 10:
                # reward_relax += 8
                # reward_relax += min(0.06 * relax_count * relax_count, 10) #二次方
                reward_relax += min(0.8 * relax_count, 11)  # 线性

        relax_count = 0
        tx_count = 0
        # 7，8，17，18时段
        if s_T == 7:
            for b in range(15):

                if s_Location[b] == 0:  # 判断无人机位置
                    tx_count += 1
                    if s_Battery[b] <= 0:  # 软约束，电池电量
                        tx_count -= 1
                if s_Location[b] == 1:
                    relax_count += 1
            if tx_count < 4:
                reward_cover -= 30
            if tx_count == 4:
                reward_cover += 64
            if tx_count == 5:
                reward_cover += 69
            if tx_count >= 6:
                reward_cover += 70
            # if relax_count <  6: #6个无人机休息，给高奖励
            #     reward_relax -= 20
            if relax_count >= 6:
                # reward_relax += 20
                # reward_relax += min(0.25 * relax_count * relax_count, 20)
                reward_relax += min(0.8 * relax_count, 8)
        if s_T == 18:
            for b in range(15):

                if s_Location[b] == 0:  # 判断无人机位置
                    tx_count += 1
                    if s_Battery[b] <= 0:  # 软约束，电池电量
                        tx_count -= 1
                if s_Location[b] == 1:
                    relax_count += 1
            if tx_count < 4:
                reward_cover -= 30
            if tx_count == 4:
                reward_cover += 56
            if tx_count == 5:
                reward_cover += 59
            if tx_count >= 6:
                reward_cover += 60
            # if relax_count <  6: #6个无人机休息，给高奖励
            #     reward_relax -= 20
            if relax_count >= 6:
                # reward_relax += 20
                # reward_relax += min(0.25 * relax_count * relax_count, 20)
                reward_relax += min(0.8 * relax_count, 8)

        if s_T == 8 or s_T == 17:
            for b in range(15):

                if s_Location[b] == 0:  # 判断无人机位置
                    tx_count += 1
                    if s_Battery[b] <= 0:  # 软约束，电池电量
                        tx_count -= 1
                if s_Location[b] == 1:
                    relax_count += 1
            if tx_count < 4:
                reward_cover -= 30
            if tx_count == 4:
                reward_cover += 83
            if tx_count == 5:
                reward_cover += 86
            if tx_count == 6:
                reward_cover += 88
            if tx_count >= 7:
                reward_cover += 90
            # if relax_count <  6: #6个无人机休息，给高奖励
            #     reward_relax -= 20
            if relax_count >= 5:
                # reward_relax += 20
                # reward_relax += min(0.35 * relax_count * relax_count, 22)
                reward_relax += min(0.8 * relax_count, 7)
        # 1个和充电电量相关，1个和覆盖率，1个坠毁相关
        # reward有正有负，正的可能是假好，尽量保证正负值域接近

        relax_count = 0
        tx_count = 0
        # 9，10，13-16时段
        if s_T == 9 or s_T == 10:
            for c in range(15):
                if s_Location[c] == 0:
                    tx_count += 1
                    if s_Battery[c] <= 0:
                        tx_count -= 1
                if s_Location[c] == 1:
                    relax_count += 1
            if tx_count < 6:
                reward_cover -= 30
            if tx_count == 6:
                reward_cover += 110
            if tx_count == 7:
                reward_cover += 114
            if tx_count == 8:
                reward_cover += 117
            if tx_count >= 9:
                reward_cover += 120
            # if relax_count < 3: # 四个无人机去充电给高奖励
            #     reward_solar -= 120
            if relax_count >= 3:
                # reward_solar += 80
                # reward_relax += min(2.23 * relax_count * relax_count, 80)
                # reward_relax += min(13.34 * relax_count, 80)
                reward_solar += min(2.5 * relax_count, 15)
        tx_count = 0
        relax_count = 0
        if s_T >= 13 and s_T <= 15:
            for d in range(15):
                if s_Location[d] == 0:
                    tx_count += 1
                    if s_Battery[d] <= 0:
                        tx_count -= 1
                if s_Location[d] == 1:
                    relax_count += 1
            if tx_count < 6:
                reward_cover -= 30
            if tx_count == 6:
                reward_cover += 110
            if tx_count == 7:
                reward_cover += 114
            if tx_count == 8:
                reward_cover += 117
            if tx_count >= 9:
                reward_cover += 120
            if relax_count >= 3:
                # reward_solar += 80
                # reward_relax += min(2.23 * relax_count * relax_count, 80)
                # reward_relax += min(13.34 * relax_count, 80)
                reward_solar += min(2.5 * relax_count, 15)

        tx_count = 0
        relax_count = 0
        if s_T == 16:
            for d in range(15):
                if s_Location[d] == 0:
                    tx_count += 1
                    if s_Battery[d] <= 0:
                        tx_count -= 1
                if s_Location[d] == 1:
                    relax_count += 1
            if tx_count < 6:
                reward_cover -= 30
            if tx_count == 6:
                reward_cover += 110
            elif tx_count == 7:
                reward_cover += 114
            elif tx_count == 8:
                reward_cover += 117
            elif tx_count >= 9:
                reward_cover += 120
            if relax_count >= 3:
                # reward_relax += 30
                # reward_relax += min(0.84 * relax_count * relax_count, 30)
                # reward_relax += min(5 * relax_count, 30)
                reward_relax += min(0.8 * relax_count, 5)
        tx_count = 0
        relax_count = 0

        #  11,12 时段
        if s_T == 11 or s_T == 12:
            for e in range(15):
                if s_Location[e] == 0:
                    tx_count += 1
                    if s_Battery[e] <= 0:
                        tx_count -= 1
                if s_Location[e] == 1:
                    relax_count += 1
            if tx_count < 5:
                reward_cover -= 30
            if tx_count == 5:
                reward_cover += 88
            elif tx_count == 6:
                reward_cover += 95
            elif tx_count == 7:
                reward_cover += 98
            elif tx_count >= 8:
                reward_cover += 100
            if relax_count >= 4:
                # reward_solar += 80
                # reward_relax += min(1.7 * relax_count * relax_count, 83)
                # reward_relax += min(11.43 * relax_count, 80)
                # reward_relax += min(13.34 * relax_count, 80)
                reward_solar += min(2.5 * relax_count, 18)
        tx_count = 0
        relax_count = 0
        #  6,19,20 时段
        if s_T == 19 or s_T == 20 or s_T == 6 :
            for f in range(15):
                if s_Location[f] == 0:
                    tx_count += 1
                    if s_Battery[f] <= 0:
                        tx_count -= 1
                if s_Location[f] == 1:
                    relax_count += 1
            if tx_count < 2:
                reward_cover -= 30
            if tx_count == 2:
                reward_cover += 26
            elif tx_count == 3:
                reward_cover += 29
            elif tx_count >= 4:
                reward_cover += 30
            if relax_count >= 8:
                # reward_relax += 20
                # reward_relax += min(0.17 * relax_count * relax_count, 20)
                # reward_relax += min(1.82 * relax_count, 20)
                reward_relax += min(0.8 * relax_count, 9)
        # if s_T == 6:
        #     for f in range(15):
        #         if s_Location[f] == 0:
        #             tx_count += 1
        #             if s_Battery[f] <= 0:
        #                 tx_count -= 1
        #         if s_Location[f] == 1:
        #             relax_count += 1
        #     if tx_count < 2:
        #         reward_cover -= 30
        #     if tx_count == 2:
        #         reward_cover += 26
        #     elif tx_count == 3:
        #         reward_cover += 29
        #     elif tx_count >= 4:
        #         reward_cover += 30
        #     if relax_count >= 8:
        #         # reward_relax += 20
        #         # reward_relax += min(0.17 * relax_count * relax_count, 20)
        #         reward_relax += min(5 * relax_count, 60)
        tx_count = 0
        relax_count = 0
        # 21时段
        # if s_T == 21:
        #     for f in range(10):
        #         if s_Location[f]==0:
        #             tx_count+=1
        #     if tx_count < 4:
        #         reward_sigle -=100
        #     if tx_count == 4:
        #         reward_sigle += 93*3
        #     elif tx_count >=5:
        #         reward_sigle += 98*3
        # tx_count = 0
        # 22，23 时段

        if s_T == 22 or s_T == 23:
            for g in range(15):
                if s_Location[g] == 0:
                    tx_count += 1
                    if s_Battery[g] <= 0:
                        tx_count -= 1
            if tx_count < 2:
                reward_sigle += 0
            if tx_count == 2:
                reward_sigle += 0
            elif tx_count >= 3:
                reward_sigle += 0
        # 0-4 时段
        if s_T >= 0 and s_T <= 4:
            for h in range(15):
                if s_Location[h] == 0:
                    tx_count += 1
                    if s_Battery[h] <= 0:
                        tx_count -= 1
            if tx_count < 2:
                reward_sigle += 0
            if tx_count == 2:
                reward_sigle += 0
            elif tx_count >= 3:
                reward_sigle += 0

        reward_sigle = reward_cover + reward_solar + reward_relax + reward_punish

        # r1:用户，print；
        # r2：
        # r3：
        # delta = reward_sigle - self.ori_reward
        # self.ori_reward = reward_sigle

        if self.count >= 18:

            done = True
            s_0 = 'terminal'
            s_1 = ''
            s_2 = ''
            s_3 = ''
            s_4 = ''
            s_5 = ''
            s_6 = ''
            s_7 = ''
            s_8 = ''
            s_9 = ''
            s_10 = ''
            s_11 = ''
            s_12 = ''
            s_13 = ''
            s_14 = ''
        # if UAVcount >= 5:
        #
        #     done = True
        #     s_0 ='terminal'
        #     s_1 =''
        #     s_2 =''
        #     s_3 =''
        #     s_4 =''
        #     s_5 =''
        #     s_6 =''
        #     s_7 =''
        #     s_8 =''
        #     s_9 =''
        #     s_10 =''
        #     s_11 =''
        #     s_12 =''
        #     s_13 =''
        #     s_14 =''
        #
        # episode定长会好训练一点
        # # # 判断电量
        # elif self.rect0[0]<0 and self.rect1[0]<0 and self.rect2[0]<0 and self.rect3[0]<0 and self.rect4[0]<0 and self.rect5[0]<0 and self.rect6[0]<0 and self.rect7[0]<0 and self.rect8[0]<0 and self.rect9[0]<0 and self.rect10[0]<0 and self.rect11[0]<0 and self.rect12[0]<0 and self.rect13[0]<0 and self.rect14[0]<0:
        #
        #     done = True
        #     s_0 = 'terminal'
        #     s_1 = ''
        #     s_2 = ''
        #     s_3 = ''
        #     s_4 = ''
        #     s_5 = ''
        #     s_6 = ''
        #     s_7 = ''
        #     s_8 = ''
        #     s_9 = ''
        #     s_10 = ''
        #     s_11 = ''
        #     s_12 = ''
        #     s_13 = ''
        #     s_14 = ''
        #     # s_0 = 'terminal'
        #     # s_1 = ''
        #     # s_2 = ''
        else:

            done = False
        print('cover:', reward_cover)
        # print('relax', reward_relax)
        # print('solar', reward_solar)
        print('UAV:', UAVcount)
        print('Battery', s_Battery)
        # print(reward_solar)
        # return s_0+s_1, delta, done
        # return list(self.rect0)+list(self.rect1)+list(self.rect2)+list(self.rect3)+list(self.rect4)+list(self.rect5)+list(self.rect6)+list(self.rect7), reward_sigle, done
        return s_0 + s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 + s_9 + s_10 + s_11 + s_12 + s_13 + s_14, reward_sigle, done