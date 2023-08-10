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
        action_probs = self.actor_model(state)
        action = np.random.choice(range(10), p=action_probs.numpy()[0])
        return action


class ActorModel(Model):
    def __init__(self):
        super(ActorModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(10, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)


def train_model(num_episodes):
    actor_model = ActorModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for episode in range(num_episodes):
        # 获得trajectory
        # 计算累积rewards
        # 训练actor model
        actor_model.save_weights('./weights.h5')

    return actor_model


def play_game(player, ai, rounds):
    player_wins = 0
    ai_wins = 0

    for episode in range(rounds):
        print(f"第{episode + 1}轮开始:")

        player_gesture, player_number = player.get_input()

        ai_action = ai.guess(player_gesture, player_number)
        ai_gesture, ai_number = ai_action // 10, ai_action % 10

        print(f"{player.name}的数字:{player_gesture}, {player_number}")
        print(f"{ai.name}的数字:{ai_gesture}, {ai_number}")

        total = player_gesture + ai_gesture
        if total == player_number:
            print(f"{player.name}胜利!")
            player_wins += 1
        elif total == ai_number:
            print(f"{ai.name}胜利!")
            ai_wins += 1
        else:
            print("平局!")

    return player_wins, ai_wins


def main():
    print("欢迎来到猜码游戏!")

    player = Player('玩家')
    ai = AI('AI', ActorModel())

    train_new_model = True
    if train_new_model:
        actor_model = train_model(100)
        ai.actor_model = actor_model
    else:
        try:
            ai.actor_model.load_weights('./weights.h5')
        except:
            ai.actor_model = ActorModel()

    rounds = int(input("请输入游戏轮数:"))
    player_wins, ai_wins = play_game(player, ai, rounds)

    print("游戏结束!")
    print(f"{player.name}胜利次数:{player_wins}")
    print(f"{ai.name}胜利次数:{ai_wins}")


if __name__ == "__main__":
    main()