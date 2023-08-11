import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
class Player:
    def __init__(self, name):
        self.name = name

    def get_input(self):
        gesture = int(input(f"{self.name}请输入您要比划的数字(0-9):"))
        while gesture < 0 or gesture > 9:
            gesture = int(input(f"{self.name}请输入有效的数字(0-9):"))

        number = int(input(f"{self.name}请输入您要说出的数字(0-18,不可以是5):"))
        while number < 0 or number > 18 or number == 5:
            number = int(input(f"{self.name}请输入有效的数字(0-18,不可以是5):"))
        return gesture, number


class AI(Player):
    def __init__(self, name, actor_model):
        super().__init__(name)
        self.actor_model = actor_model

    def guess(self, player_gesture, player_number):
        state = np.array([[player_gesture, player_number]])
        gesture_probs, number_probs = self.actor_model(state)

        max_gesture = np.max(gesture_probs)
        gesture_probs = np.exp(gesture_probs - max_gesture) / np.sum(np.exp(gesture_probs - max_gesture))

        max_number = np.max(number_probs)
        number_probs = np.exp(number_probs - max_number) / np.sum(np.exp(number_probs - max_number))

        gesture = np.random.choice(10, p=gesture_probs[0])
        number = np.random.choice(19, p=number_probs[0])

        return gesture, number


class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.gesture_out = Dense(10, activation='softmax')
        self.number_out = Dense(19, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        gesture_probs = self.gesture_out(x)
        number_probs = tf.nn.softmax(self.number_out(x))  # 对number_out进行softmax操作
        return gesture_probs, number_probs


def train_model(num_episodes):
    actor_model = ActorModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for episode in range(num_episodes):
        # 获得trajectory
        # 计算累积rewards
        # 训练actor model
        actor_model.save_weights('./model.h5')

    return actor_model


def play_game(player, ai, rounds):
    player_wins = 0
    ai_wins = 0

    for episode in range(rounds):

        print(f"第{episode + 1}轮开始:")

        # 玩家输入
        player_gesture, player_number = player.get_input()

        # AI猜测
        ai_gesture, ai_number = ai.guess(player_gesture, player_number)

        # 输出双方选择
        print(f"{player.name}的数字:{player_gesture}, {player_number}")
        print(f"{ai.name}的数字:{ai_gesture}, {ai_number}")

        # 判断胜负
        total_1 = player_gesture + ai_gesture
        #total_2 = player_number + ai_number
        if total_1 == player_gesture:
            print(f"{player.name}胜利!")
            player_wins += 1
        elif total_1 == ai_gesture:
            print(f"{ai.name}胜利!")
            ai_wins += 1
        else:
            print("平局!")

    return player_wins, ai_wins


def main():
    #ai = AI('AI')
    print("欢迎来到猜码游戏!")

    player = Player('玩家')
    ai = AI('AI', ActorModel())

    train_new_model = True
    if train_new_model:
        actor_model = train_model(100)
        ai.actor_model = actor_model
    else:
        try:
            ai.actor_model.load_weights('./model.h5')
        except:
            ai.actor_model = ActorModel()

    rounds = int(input("请输入游戏轮数:"))
    player_wins, ai_wins = play_game(player, ai, rounds)

    print("游戏结束!")
    print(f"{player.name}胜利次数:{player_wins}")
    print(f"{ai.name}胜利次数:{ai_wins}")


if __name__ == "__main__":
    main()