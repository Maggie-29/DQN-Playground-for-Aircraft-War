# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:05:00 2013

@author: Leo

Modified on Sat Jun 20 18:57:55 2020

@modifier: Karbon

"""

import pygame
from sys import exit
from pygame.locals import *
import random
import time
import numpy as np


SCREEN_WIDTH = 480
SCREEN_HEIGHT = 800

TYPE_SMALL = 1
TYPE_MIDDLE = 2
TYPE_BIG = 3

# 子弹类
class Bullet(pygame.sprite.Sprite):
    def __init__(self, bullet_img, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = bullet_img
        self.rect = self.image.get_rect()
        self.rect.midbottom = init_pos
        self.speed = 10

    def move(self):
        self.rect.top -= self.speed

# 玩家类
class Player(pygame.sprite.Sprite):
    def __init__(self, plane_img, player_rect, init_pos):
        pygame.sprite.Sprite.__init__(self)
        self.image = []                                 # 用来存储玩家对象精灵图片的列表
        for i in range(len(player_rect)):
            self.image.append(plane_img.subsurface(player_rect[i]).convert_alpha())
        self.rect = player_rect[0]                      # 初始化图片所在的矩形
        self.rect.topleft = init_pos                    # 初始化矩形的左上角坐标
        self.speed = 8                                  # 初始化玩家速度，这里是一个确定的值
        self.bullets = pygame.sprite.Group()            # 玩家飞机所发射的子弹的集合
        self.img_index = 0                              # 玩家精灵图片索引
        self.is_hit = False                             # 玩家是否被击中

    def shoot(self, bullet_img):
        bullet = Bullet(bullet_img, self.rect.midtop)
        self.bullets.add(bullet)

    def moveUp(self):
        if self.rect.top <= 0:
            self.rect.top = 0
        else:
            self.rect.top -= self.speed

    def moveDown(self):
        if self.rect.top >= SCREEN_HEIGHT - self.rect.height:
            self.rect.top = SCREEN_HEIGHT - self.rect.height
        else:
            self.rect.top += self.speed

    def moveLeft(self):
        if self.rect.left <= 0:
            self.rect.left = 0
        else:
            self.rect.left -= self.speed

    def moveRight(self):
        if self.rect.left >= SCREEN_WIDTH - self.rect.width:
            self.rect.left = SCREEN_WIDTH - self.rect.width
        else:
            self.rect.left += self.speed

    def rebirth(self):  # 复活
        self.rect.topleft = [200, 600]                  # 初始化矩形的左上角坐标
        self.speed = 8                                  # 初始化玩家速度，这里是一个确定的值
        self.bullets = pygame.sprite.Group()            # 玩家飞机所发射的子弹的集合
        self.img_index = 0                              # 玩家精灵图片索引
        self.is_hit = False                             # 玩家是否被击中

# 敌人类
class Enemy(pygame.sprite.Sprite):
    def __init__(self, enemy_img, enemy_down_imgs, init_pos):
       pygame.sprite.Sprite.__init__(self)
       self.image = enemy_img
       self.rect = self.image.get_rect()
       self.rect.topleft = init_pos
       self.down_imgs = enemy_down_imgs
       self.speed = 2
       self.down_index = 0

    def move(self):
        self.rect.top += self.speed


class game:

    action_space = ['Up', 'Down', 'Left', 'Right', 'None']
    n_actions = len(action_space)
    n_features = 20

    tickSet = 60  # 设置帧数
    # 当前分数的倍数\失败了
    rewardSet = [0.001, -100000.0]

    reward = 0
    observation = np.zeros([10, 2])  # 初始化一个 10 * 2 的矩阵用于存储所有敌机的位置
    # 第一位表示敌机序号0~19
    # 第二位为0表示自己与敌机在横坐标上的差，为1表示纵坐标上的差

    done = False

    stepping = False

    # 初始化游戏
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('飞机大战')

    # 载入游戏音乐
    bullet_sound = pygame.mixer.Sound('resources/sound/bullet.wav')
    enemy1_down_sound = pygame.mixer.Sound('resources/sound/enemy1_down.wav')
    game_over_sound = pygame.mixer.Sound('resources/sound/game_over.wav')
    bullet_sound.set_volume(0.3)
    enemy1_down_sound.set_volume(0.3)
    game_over_sound.set_volume(0.3)
    pygame.mixer.music.load('resources/sound/game_music.wav')
    pygame.mixer.music.play(-1, 0.0)
    pygame.mixer.music.set_volume(0.25)

    # 载入背景图
    background = pygame.image.load('resources/image/background.png').convert()
    game_over = pygame.image.load('resources/image/gameover.png')

    filename = 'resources/image/shoot.png'
    plane_img = pygame.image.load(filename)

    # 设置玩家相关参数
    player_rect = []
    player_rect.append(pygame.Rect(0, 99, 102, 126))        # 玩家精灵图片区域
    player_rect.append(pygame.Rect(165, 360, 102, 126))
    player_rect.append(pygame.Rect(165, 234, 102, 126))     # 玩家爆炸精灵图片区域
    player_rect.append(pygame.Rect(330, 624, 102, 126))
    player_rect.append(pygame.Rect(330, 498, 102, 126))
    player_rect.append(pygame.Rect(432, 624, 102, 126))
    player_pos = [200, 600]
    player = Player(plane_img, player_rect, player_pos)

    # 定义子弹对象使用的surface相关参数
    bullet_rect = pygame.Rect(1004, 987, 9, 21)
    bullet_img = plane_img.subsurface(bullet_rect)

    # 定义敌机对象使用的surface相关参数
    enemy1_rect = pygame.Rect(534, 612, 57, 43)
    enemy1_img = plane_img.subsurface(enemy1_rect)
    enemy1_down_imgs = []
    enemy1_down_imgs.append(plane_img.subsurface(
        pygame.Rect(267, 347, 57, 43)))
    enemy1_down_imgs.append(plane_img.subsurface(
        pygame.Rect(873, 697, 57, 43)))
    enemy1_down_imgs.append(plane_img.subsurface(
        pygame.Rect(267, 296, 57, 43)))
    enemy1_down_imgs.append(plane_img.subsurface(
        pygame.Rect(930, 697, 57, 43)))

    enemies1 = pygame.sprite.Group()

    # 存储被击毁的飞机，用来渲染击毁精灵动画
    enemies_down = pygame.sprite.Group()

    shoot_frequency = 0
    enemy_frequency = 0

    player_down_index = 16

    score = 0

    clock = pygame.time.Clock()

    running = True

    def step(self, action):
        self.stepping = True

        sthHappened = False
        self.observation = np.zeros([10, 2])

        # 控制游戏最大帧率为60
        self.clock.tick(self.tickSet)

        # 控制发射子弹频率,并发射子弹
        if not self.player.is_hit:
            if self.shoot_frequency % 15 == 0:
                self.bullet_sound.play()
                self.player.shoot(self.bullet_img)
            self.shoot_frequency += 1
            if self.shoot_frequency >= 15:
                self.shoot_frequency = 0

        # 生成敌机
        if self.enemy_frequency % 50 == 0:
            enemy1_pos = [random.randint(
                0, SCREEN_WIDTH - self.enemy1_rect.width), 0]
            enemy1 = Enemy(self.enemy1_img, self.enemy1_down_imgs, enemy1_pos)
            self.enemies1.add(enemy1)
        self.enemy_frequency += 1
        if self.enemy_frequency >= 100:
            self.enemy_frequency = 0

        # 移动子弹，若超出窗口范围则删除
        for bullet in self.player.bullets:
            bullet.move()
            if bullet.rect.bottom < 0:
                self.player.bullets.remove(bullet)

        # 移动敌机，若超出窗口范围则删除
        for enemy in self.enemies1:
            enemy.move()
            # 判断玩家是否被击中
            if pygame.sprite.collide_circle(enemy, self.player):
                self.enemies_down.add(enemy)
                self.enemies1.remove(enemy)
                self.player.is_hit = True
                self.game_over_sound.play()
                self.reward = self.rewardSet[1]  # 被击中
                sthHappened = True
                break
            if enemy.rect.top > SCREEN_HEIGHT:
                self.enemies1.remove(enemy)
                sthHappened = True

        # 将被击中的敌机对象添加到击毁敌机Group中，用来渲染击毁动画
        enemies1_down = pygame.sprite.groupcollide(
            self.enemies1, self.player.bullets, 1, 1)
        for enemy_down in enemies1_down:
            self.enemies_down.add(enemy_down)

        # 绘制背景
        self.screen.fill(0)
        self.screen.blit(self.background, (0, 0))

        # 绘制玩家飞机
        if not self.player.is_hit:
            self.screen.blit(
                self.player.image[self.player.img_index], self.player.rect)
            # 更换图片索引使飞机有动画效果
            self.player.img_index = self.shoot_frequency // 8
        else:
            self.player.img_index = self.player_down_index // 8
            self.screen.blit(
                self.player.image[self.player.img_index], self.player.rect)
            self.player_down_index += 1
            if self.player_down_index > 47:
                self.running = False
                self.plane_down()

        # 绘制击毁动画
        for enemy_down in self.enemies_down:
            if enemy_down.down_index == 0:
                self.enemy1_down_sound.play()
            if enemy_down.down_index > 7:
                self.enemies_down.remove(enemy_down)
                self.score += 1000
                sthHappened = True
                continue
            self.screen.blit(
                enemy_down.down_imgs[enemy_down.down_index // 2], enemy_down.rect)
            enemy_down.down_index += 1

        # 绘制子弹和敌机
        self.player.bullets.draw(self.screen)
        self.enemies1.draw(self.screen)

        # 绘制得分
        score_font = pygame.font.Font(None, 36)
        score_text = score_font.render(str(self.score), True, (128, 128, 128))
        text_rect = score_text.get_rect()
        text_rect.topleft = [10, 10]
        self.screen.blit(score_text, text_rect)

        # 更新屏幕
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # 监听键盘事件
        key_pressed = pygame.key.get_pressed()
        # 若玩家被击中，则无效
        if not self.player.is_hit:
            if key_pressed[K_w] or key_pressed[K_UP] or action == 0:
                self.player.moveUp()
            if key_pressed[K_s] or key_pressed[K_DOWN] or action == 1:
                self.player.moveDown()
            if key_pressed[K_a] or key_pressed[K_LEFT] or action == 2:
                self.player.moveLeft()
            if key_pressed[K_d] or key_pressed[K_RIGHT] or action == 3:
                self.player.moveRight()

        i = 0
        for enemy in self.enemies1:
            self.observation[i] = [(self.player.rect.centerx - enemy.rect.centerx) / 480.0,
                (self.player.rect.centery - enemy.rect.centery) / 800.0]
            i = i + 1

        for j in range(i, 10):
            self.observation[j] = [0., 1.]
        
        self.observation = self.observation[np.argsort(self.observation[:,1])]
        self.observation = self.observation.flatten()

        #XXX: if not sthHappened:
        if True:
            if not self.done:
                self.reward = self.rewardSet[0] * self.score
            else:
                self.reward = self.rewardSet[1]
        
        # time.sleep(0.1)
        # print('\f' + self.observation)
        # print("%6.4f\t%s" %
        #       (self.reward, (lambda x: 'True' if x else 'False')(self.done)))

        self.stepping = False

        return self.observation, self.reward, self.done

    def restart(self):
        thisScore = self.score
        self.enemies1 = pygame.sprite.Group()
        self.enemies_down = pygame.sprite.Group()
        self.shoot_frequency = 0
        self.enemy_frequency = 0
        self.player_down_index = 16
        self.score = 0
        self.clock = pygame.time.Clock()
        self.running = True
        self.done = False
        self.player.rebirth()
        return thisScore

    def plane_down(self):
        font = pygame.font.Font(None, 48)
        text = font.render('Score: ' + str(self.score), True, (255, 0, 0))
        text_rect = text.get_rect()
        text_rect.centerx = self.screen.get_rect().centerx
        text_rect.centery = self.screen.get_rect().centery + 24
        self.screen.blit(self.game_over, (0, 0))
        self.screen.blit(text, text_rect)
        self.done = True

    def set_tick(self, num):
        self.tickSet = num

    def set_reward(self, num0, num1):
        self.rewardSet = [num0, num1]
        return

        

if __name__ == "__main__":
    game0 = game()
    while(True):
        game0.step("None")
        if game0.done == True:
            game0.restart()
