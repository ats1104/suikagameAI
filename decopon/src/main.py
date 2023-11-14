import random
from collections import namedtuple
from typing import Tuple

import pygame
import pygame.surfarray
import pymunk

import gym
from gym.spaces import Box, Discrete
from gym.wrappers import FrameStack
from gym.envs.registration import register

import numpy as np
from PIL import Image
import cv2


from decopon.controller import Controller, Human, AIPlayer, MetricLogger

Polygon = namedtuple("Polygon", ["mass", "radius", "color", "score", "index"])
Polygons = [
    Polygon(1, 10, (255, 0, 127), 0, 0),
    Polygon(2, 20, (255, 0, 255), 1, 1),
    Polygon(3, 30, (127, 0, 255), 3, 2),
    Polygon(4, 40, (0, 0, 255), 6, 3),
    Polygon(5, 50, (0, 127, 255), 10, 4),
    Polygon(6, 60, (0, 255, 255), 15, 5),
    Polygon(7, 70, (0, 255, 127), 21, 6),
    Polygon(8, 80, (0, 255, 0), 28, 7),
    Polygon(9, 90, (127, 255, 0), 36, 8),
    Polygon(10, 100, (255, 255, 0), 45, 9),
    Polygon(11, 110, (255, 127, 0), 55, 10),
]

pygame.init()
pygame.display.set_caption("Decopon")

HEIGHT, WIDTH = 640, 480
TIMELIMIT = 1000

INPUT = (3, 84, 84)
PATH = 'result'

class Game:
    def __init__(self, controller: Controller):
       
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.fps = pygame.time.Clock().tick
        self.controller = controller

        self.space = pymunk.Space()
        self.space.gravity = (0, 1000)
        collision_handler = self.space.add_collision_handler(1, 1)
        collision_handler.begin = self.merge

        self.walls = []
        self.create_walls()

        self.indicator = pygame.Rect(WIDTH / 2, 200, 3, HEIGHT - 200)

        self.poly = []

        self.drop_ticks = pygame.time.get_ticks()

        self.current = random.randint(0, 4)
        self.next = random.randint(0, 4)

        self.font = pygame.font.Font("resources/BestTen-DOT.otf", 16)
        self.score = 0

        self.isGameOver = False
        self.countOverflow = 0

        self.progress = [pygame.Rect(10 + i * 20, 70, 20, 20) for i in range(11)]

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 1000)
        collision_handler = self.space.add_collision_handler(1, 1)
        collision_handler.begin = self.merge
        self.walls = []
        self.create_walls()

        self.indicator = pygame.Rect(WIDTH / 2, 200, 3, HEIGHT - 200)

        self.poly = []

        self.drop_ticks = pygame.time.get_ticks()

        self.current = random.randint(0, 4)
        self.next = random.randint(0, 4)

        self.font = pygame.font.Font("resources/BestTen-DOT.otf", 16)
        self.score = 0

        self.isGameOver = False
        self.countOverflow = 0
        self.progress = [pygame.Rect(10 + i * 20, 70, 20, 20) for i in range(11)]   
    def merge(self, polys, space, _):
        p0, p1 = polys.shapes

        if p0.index == 10 and p1.index == 10:
            self.poly.remove(p0)
            self.poly.remove(p1)
            space.remove(p0, p0.body, p1, p1.body)
            # ここ何点？
            self.score += 1000
            return False

        if p0.index == p1.index:
            self.score += Polygons[p0.index].score
            x = (p0.body.position.x + p1.body.position.x) / 2
            y = (p0.body.position.y + p1.body.position.y) / 2
            self.create_poly(x, y, p1.index + 1)
            self.poly.remove(p0)
            self.poly.remove(p1)
            space.remove(p0, p0.body, p1, p1.body)

        return True

    def create_walls(self):
        floor = pymunk.Segment(self.space.static_body, (0, HEIGHT - 10), (WIDTH, HEIGHT - 10), 10)
        floor.friction = 0.8
        floor.elasticity = 0.8

        self.walls.append(floor)
        self.space.add(floor)

        left = pymunk.Segment(self.space.static_body, (10, 300), (10, HEIGHT), 10)
        left.friction = 1.0
        left.elasticity = 0.95
        self.walls.append(left)
        self.space.add(left)

        right = pymunk.Segment(self.space.static_body, (WIDTH - 10, 300), (WIDTH - 10, HEIGHT), 10)
        right.friction = 1.0
        right.elasticity = 0.95
        self.walls.append(right)
        self.space.add(right)

        self.start_time = pygame.time.get_ticks()

    def check_event(self, event):
        for e in pygame.event.get():
            if e.type == event:
                return True
        return False

    def create_poly(self, x: int, y: int, idx: int):
        poly = Polygons[idx]

        mass = poly.mass
        radius = poly.radius
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        body.radius = radius
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        shape.elasticity = 0.4
        shape.collision_type = 1
        shape.index = poly.index
        shape.color = poly.color
        self.space.add(body, shape)
        self.poly.append(shape)

    def draw_walls(self):
        pygame.draw.line(self.window, (240, 190, 153), (0, 300), (WIDTH, 300), 5)
        for wall in self.walls:
            p1 = int(wall.a[0]), int(wall.a[1])
            p2 = int(wall.b[0]), int(wall.b[1])
            pygame.draw.line(self.window, (240, 190, 153), p1, p2, 20)

    def check_overflow(self):
        for poly in self.poly:
            if poly.body.position.y - (poly.radius) < 300:
                return True
        return False

   

    def run(self):

        while True:

            array = pygame.surfarray.array3d(self.window)
            image = Image.fromarray(np.uint8(array))
            # 画像のサイズを変更
            resized_image = image.resize((84, 84), Image.ANTIALIAS)
            # PIL ImageをNumPy配列に戻す
            state = np.array(resized_image)

            seconds = (pygame.time.get_ticks() - self.start_time) // 1000

            if self.isGameOver or seconds > TIMELIMIT:
                print(self.score)
                # self._setup()

                exit(0)

            if self.check_event(pygame.QUIT):
                break
            if self.check_overflow():
                self.countOverflow += 1

            isLeft, isRight, isDrop = self.controller.update(state)

            if isLeft:
                self.indicator.centerx -= 3
            elif isRight:
                self.indicator.centerx += 3
            elif isDrop and pygame.time.get_ticks() - self.drop_ticks > 500 and not self.check_overflow():
                self.create_poly(self.indicator.centerx, self.indicator.topleft[1], self.current)
                self.drop_ticks = pygame.time.get_ticks()
                # 次の果物を現在のものにセット．
                self.current = self.next
                # 次の果物を生成
                self.next = random.randint(0, 4)
                self.countOverflow = 0

            if self.indicator.centerx < 20:
                self.indicator.centerx = WIDTH - 20
            if self.indicator.centerx > WIDTH - 20:
                self.indicator.centerx = 20

            self.window.fill((89, 178, 36))
            pygame.draw.rect(self.window, (255, 255, 255), self.indicator)

            poly = Polygons[self.current]
            pygame.draw.circle(
                self.window, poly.color, (self.indicator.centerx, self.indicator.topleft[1]), poly.radius
            )
            poly = Polygons[self.next]
            pygame.draw.circle(self.window, poly.color, (WIDTH - 60, 60), poly.radius)

            for poly in self.poly:
                pygame.draw.circle(
                    self.window, poly.color, (int(poly.body.position.x), int(poly.body.position.y)), poly.radius
                )

            self.draw_walls()

            score_text = self.font.render(f"スコア: {self.score}", True, (255, 255, 255))
            score_position = (10, 10)
            self.window.blit(score_text, score_position)

            text = self.font.render(f"残り時間: {TIMELIMIT - seconds}", True, (255, 255, 255))
            position = (10, 30)
            self.window.blit(text, position)

            text = self.font.render("シンカ", True, (255, 255, 255))
            position = (10, 50)
            self.window.blit(text, position)

            for i, poly in enumerate(self.progress):
                pygame.draw.rect(self.window, Polygons[i].color, poly)

            self.space.step(1 / 60)
            pygame.display.update()
            # 画面の画像情報の取得
            # (480, 640, 3)




            if self.countOverflow > 200:
                self.isGameOver = True

class MyEnv(gym.Env, Game):

    """"
    must methods -----
    reset:環境の初期化，初期状態のを観測して返す
    step:行動を受け取り, 行動後のstateのobservation, reward, done, infoを返す
    render:描画, 配列のreturn
    -----
    
    observation:object型
    reward:float型
    done:bool型
    info:dict型
    """

    # エージェントの行動空間，観測空間，報酬の範囲

    def __init__(self, test=False):

        super().__init__(AIPlayer(INPUT, 3, f'./{PATH}', test))
        self.action_space = Discrete(3)

        self.observation_space = Box(0, 255, (84, 84, 3))
        self.reward_range = (-1, 1)

        self.update_counter = 0
        self.frame_num = 0

        self.epsode_num = 0
        # 最後に地面にいた時の報酬
        self.frame_integration = 10
        self.logger = MetricLogger(f'./{PATH}')
        self.test = test # bool
        self.prev_step = 0
        
        self.reward = 0

        self.reset()
 

    def _setup(self):
        
        self.space = pymunk.Space()
        self.space.gravity = (0, 1000)
        collision_handler = self.space.add_collision_handler(1, 1)
        collision_handler.begin = self.merge
        self.walls = []
        self.create_walls()

        self.indicator = pygame.Rect(WIDTH / 2, 200, 3, HEIGHT - 200)

        self.poly = []

        self.drop_ticks = pygame.time.get_ticks()

        self.current = random.randint(0, 4)
        self.next = random.randint(0, 4)

        self.font = pygame.font.Font("resources/BestTen-DOT.otf", 16)
        self.score = 0

        self.isGameOver = False
        self.countOverflow = 0
        self.progress = [pygame.Rect(10 + i * 20, 70, 20, 20) for i in range(11)]   
        self.seconds = 0
        self.done = False

    # rest 状態の初期化と初期状態の観測，
    def reset(self):
        # ゲームの初期化
        self._setup()
        # self.controller.curr_step = 0
        return self.get_obs()

    # 行動を受け取り行動後の環境状態の観測，その時の報酬，エピソード終了判定を返す．
    def step(self, action):
        # 4フレーム処理を進める．
        total_reward = 0
        prev_score = self.score
        for _ in range(self.frame_integration):
            self.update_game(action)
            total_reward += self.get_reward(prev_score)
            prev_score = self.score
            self.check_done()
            if self.done:
                break
        # 4 フレーム後の状態を取得
        next_state = self.get_obs()
        
        info = {}
        # 報酬は4フレーム分まとめて．

        return next_state, total_reward, self.done, info
        
    def render(self):
        pass

    def get_obs(self):
        # 画像を受け取って，ndarrayとして取得
        array = pygame.surfarray.array3d(self.window)
        image = Image.fromarray(np.uint8(array))
        # 画像のサイズを変更
        resized_image = image.resize((84, 84), Image.ANTIALIAS)
        # PIL ImageをNumPy配列に戻す
        state = np.array(resized_image)


        return state

    def get_reward(self, prev):
        
        rw = self.score - prev
        
        return rw

    def check_done(self):
        # 一旦無限ループにはなるが．
        if self.isGameOver or self.seconds > TIMELIMIT:
            self.done = True
        
    # ゲームを1フレーム進める処理
    def update_game(self, action):

        self.seconds = (pygame.time.get_ticks() - self.start_time) // 1000

        if self.check_overflow():
            self.countOverflow += 1
        # action
        isLeft, isRight, isDrop = action

        if isLeft:
            self.indicator.centerx -= 3
        elif isRight:
            self.indicator.centerx += 3
        elif isDrop and pygame.time.get_ticks() - self.drop_ticks > 500 and not self.check_overflow():
            self.create_poly(self.indicator.centerx, self.indicator.topleft[1], self.current)
            self.drop_ticks = pygame.time.get_ticks()
            self.current = self.next
            self.next = random.randint(0, 4)
            self.countOverflow = 0

        if self.indicator.centerx < 20:
            self.indicator.centerx = WIDTH - 20
        if self.indicator.centerx > WIDTH - 20:
            self.indicator.centerx = 20

        self.window.fill((89, 178, 36))
        pygame.draw.rect(self.window, (255, 255, 255), self.indicator)

        poly = Polygons[self.current]
        pygame.draw.circle(
            self.window, poly.color, (self.indicator.centerx, self.indicator.topleft[1]), poly.radius
        )
        poly = Polygons[self.next]
        pygame.draw.circle(self.window, poly.color, (WIDTH - 60, 60), poly.radius)

        for poly in self.poly:
            pygame.draw.circle(
                self.window, poly.color, (int(poly.body.position.x), int(poly.body.position.y)), poly.radius
            )

        self.draw_walls()

        score_text = self.font.render(f"スコア: {self.score}", True, (255, 255, 255))
        score_position = (10, 10)
        self.window.blit(score_text, score_position)

        text = self.font.render(f"残り時間: {TIMELIMIT - self.seconds}", True, (255, 255, 255))
        position = (10, 30)
        self.window.blit(text, position)

        text = self.font.render("シンカ", True, (255, 255, 255))
        position = (10, 50)
        self.window.blit(text, position)

        for i, poly in enumerate(self.progress):
            pygame.draw.rect(self.window, Polygons[i].color, poly)
    
        self.space.step(1 / 60)
        pygame.display.update()

        if self.countOverflow > 200:
            self.isGameOver = True


 
       
class RandomPlayer(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self) -> Tuple[bool, bool, bool]:
        return tuple(random.choice([True, False]) for _ in range(3))

# 登録
if __name__ == "__main__":
    # Game(Human()).run()
    # AIでやる場合
    Game(AIPlayer(INPUT,  3, f'./{PATH}', True)).run()


