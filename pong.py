import pygame
import random


FPS = 5

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60

PADDLE_BUFFER = 10

BALL_WIDTH = 10
BALL_HEIGHT = 10

PADDLE_SPEED = 2

BALL_SPEED_X = 3
BALL_SPEED_Y = 2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


def draw_ball(ball_x_pos, ball_y_pos):
    pygame.draw.rect(screen, WHITE, [ball_x_pos, ball_y_pos, BALL_WIDTH, BALL_HEIGHT])


def draw_paddle1(paddle1_y_pos):
    pygame.draw.rect(screen, WHITE, [PADDLE_BUFFER, paddle1_y_pos, PADDLE_WIDTH, PADDLE_HEIGHT])


def draw_paddle2(paddle2_y_pos):
    pygame.draw.rect(screen, WHITE, [WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH,
                                     paddle2_y_pos, PADDLE_WIDTH, PADDLE_HEIGHT])


def update_ball(paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction):

    ball_x_pos += ball_x_direction * BALL_SPEED_X
    ball_y_pos += ball_y_direction * BALL_SPEED_Y
    score = 0

    if (ball_x_pos <= PADDLE_BUFFER + PADDLE_WIDTH and ball_y_pos + BALL_HEIGHT >= paddle1_y_pos and ball_y_pos - BALL_HEIGHT <= paddle1_y_pos + PADDLE_HEIGHT):
        ball_x_direction = 1
    elif ball_x_pos <= 0:
        ball_x_direction = 1
        score = -1
        return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]
    if ball_x_pos >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER and ball_y_pos + BALL_HEIGHT >= paddle2_y_pos and ball_y_pos - BALL_HEIGHT<= paddle2_y_pos + PADDLE_HEIGHT:
        ball_x_direction = -1
    elif ball_x_pos >= WINDOW_WIDTH - BALL_WIDTH:
        ball_x_direction = -1
        score = 1
        return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]
    if ball_y_pos <= 0:
        ball_y_pos = 0
        ball_y_direction = 1
    elif ball_y_pos >= WINDOW_HEIGHT - BALL_HEIGHT:
        ball_y_pos = WINDOW_HEIGHT - BALL_HEIGHT
        ball_y_direction = -1
    return [score, paddle1_y_pos, paddle2_y_pos, ball_x_pos, ball_y_pos, ball_x_direction, ball_y_direction]


def update_paddle1(action, paddle1_y_pos):
    if action[1] == 1:
        paddle1_y_pos -= PADDLE_SPEED

    if action[2] == 1:
        paddle1_y_pos += PADDLE_SPEED
    if paddle1_y_pos < 0:
        paddle1_y_pos = 0
    if paddle1_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT:
        paddle1_y_pos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle1_y_pos


def update_paddle2(paddle2_y_pos, ball_x_pos):
    if paddle2_y_pos + PADDLE_HEIGHT / 2 < ball_x_pos + BALL_HEIGHT / 2:
        paddle2_y_pos += PADDLE_SPEED

    if paddle2_y_pos + PADDLE_HEIGHT / 2 > ball_x_pos + BALL_HEIGHT / 2:
        paddle2_y_pos -= PADDLE_SPEED

    if paddle2_y_pos < 0:
        paddle2_y_pos = 0

    if paddle2_y_pos > WINDOW_HEIGHT - PADDLE_HEIGHT:
        paddle2_y_pos = WINDOW_HEIGHT - PADDLE_HEIGHT
    return paddle2_y_pos


class PongGame:
    def __init__(self):
        num = random.randint(0, 9)
        self.tally = 0
        self.paddle1_y_pos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2_y_pos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.ball_x_direction = 1
        self.ball_y_direction = 1
        self.ball_x_pos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2

        if 0 < num < 3:
            self.ballXDirection = 1
            self.ballYDirection = 1
        if 3 <= num < 5:
            self.ballXDirection = -1
            self.ballYDirection = 1
        if 5 <= num < 8:
            self.ballXDirection = 1
            self.ballYDirection = -1
        if 8 <= num < 10:
            self.ballXDirection = -1
            self.ballYDirection = -1

        num = random.randint(0, 9)

        self.ball_y_pos = num * (WINDOW_HEIGHT - BALL_HEIGHT) / 9

    def get_present_frame(self):
        pygame.event.pump()
        screen.fill(BLACK)
        draw_paddle1(self.paddle1_y_pos)
        draw_paddle2(self.paddle2_y_pos)
        draw_ball(self.ball_x_pos, self.ball_y_pos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        return image_data

    def get_next_frame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        self.paddle1_y_pos = update_paddle1(action, self.paddle1_y_pos)
        draw_paddle1(self.paddle1_y_pos)
        self.paddle2_y_pos = update_paddle2(self.paddle2_y_pos, self.ball_y_pos)
        draw_paddle2(self.paddle2_y_pos)
        [score, self.paddle1_y_pos, self.paddle2_y_pos,
         self.ball_x_pos, self.ball_y_pos, self.ball_x_direction,
         self.ball_y_direction] = update_ball(self.paddle1_y_pos, self.paddle2_y_pos, self.ball_x_pos, self.ball_y_pos,
         self.ball_x_direction, self.ball_y_direction)
        draw_ball(self.ball_x_pos, self.ball_y_pos)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.flip()
        self.tally += score

        return [score, image_data]
