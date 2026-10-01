import tkinter as tk
import numpy as np
import time

## define function
def TransMat(now_state, action):
    max_row = 4
    max_col = 12
    now_row = int(now_state/max_col)
    now_col = (now_state%max_col)

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:
        print('index error')
        return

    col = now_col
    row = now_row
    if action == 0 and now_row > 0:
        row -= 1
    elif action == 1 and now_col > 0:
        col -= 1
    elif action == 2 and (max_row-1) > now_row:
        row += 1
    elif action == 3 and (max_col-1) > now_col:
        col += 1
    next_state = row*max_col + col
    return next_state

def qlearn(action_value, reward, steps, gamma, alpha, epsilon):
    record = []
    state = 36
    fall = 0                # 記錄本回合是否掉入懸崖
    for step in range(steps):
        action = GetAction(action_value, epsilon, state)
        next_state = TransMat(state, action)
        record.append([state, action, reward[next_state], next_state])
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step], alpha, gamma)

        a2 = np.around(action_value, decimals=2)

        if action == 0:
            mycanvas.delete(str(state)+'_0')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_0')
            root.update()
            mycanvas.delete(str(state)+'_0')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25, fill='#000', text=a2[state, action], tag=str(state)+'_0')
        elif action == 1:
            mycanvas.delete(str(state)+'_1')
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_1')
            root.update()
            mycanvas.delete(str(state)+'_1')
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50, fill='#000', text=a2[state, action], tag=str(state)+'_1')
        elif action == 2:
            mycanvas.delete(str(state)+'_2')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_2')
            root.update()
            mycanvas.delete(str(state)+'_2')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75, fill='#000', text=a2[state, action], tag=str(state)+'_2')
        elif action == 3:
            mycanvas.delete(str(state)+'_3')
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_3')
            root.update()
            mycanvas.delete(str(state)+'_3')
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50, fill='#000', text=a2[state, action], tag=str(state)+'_3')

        state = next_state
        if state == 47:
            print(",[success]")
            break
        elif state > 36:
            print(",[fail]")
            fall = 1            # 掉入懸崖
            break

    record = np.array(record, dtype=object)
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward, fall       # 多回傳 fall

def sarsa(action_value, reward, steps, gamma, alpha, epsilon):
    record = []
    state = 36
    fall = 0                # 記錄本回合是否掉入懸崖
    action = GetAction(action_value, epsilon, state)
    for step in range(steps):
        next_state = TransMat(state, action)
        next_action = GetAction(action_value, epsilon, next_state)
        record.append([state, action, reward[next_state], next_state, next_action])
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step], alpha, gamma)

        a2 = np.around(action_value, decimals=2)

        if action == 0:
            mycanvas.delete(str(state)+'_0')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_0')
            root.update()
            mycanvas.delete(str(state)+'_0')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25, fill='#000', text=a2[state, action], tag=str(state)+'_0')
        elif action == 1:
            mycanvas.delete(str(state)+'_1')
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_1')
            root.update()
            mycanvas.delete(str(state)+'_1')
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50, fill='#000', text=a2[state, action], tag=str(state)+'_1')
        elif action == 2:
            mycanvas.delete(str(state)+'_2')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_2')
            root.update()
            mycanvas.delete(str(state)+'_2')
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75, fill='#000', text=a2[state, action], tag=str(state)+'_2')
        elif action == 3:
            mycanvas.delete(str(state)+'_3')
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50, fill='#f00', font=('Arial', 20, 'bold'), text=a2[state, action], tag=str(state)+'_3')
            root.update()
            mycanvas.delete(str(state)+'_3')
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50, fill='#000', text=a2[state, action], tag=str(state)+'_3')

        state = next_state
        action = next_action
        if state == 47:
            print(",[success]")
            break
        elif state > 36:
            print(",[fail]")
            fall = 1            # 掉入懸崖
            break

    record = np.array(record, dtype=object)
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward, fall       # 多回傳 fall

def GetAction(action_value, epsilon, next_state):
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis=1)
        action = policy[next_state]
    else:
        action = np.random.randint(0, 4)
    return action

def ValueUpdate(method, action_value, record, alpha, gamma):
    state = record[0]
    action = record[1]
    reward = record[2]
    next_state = record[3]
    now_value = action_value[state, action]
    if method == 'qlearn':
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)
    elif method == 'sarsa':
        next_action = record[4]
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)
    else:
        print('No this method.')
        return
    value = now_value + update_value
    return value

# main
def main(episodes, method):
    ActionValue = np.zeros([48, 4])
    Reward = np.full(48, -1)
    Reward[37:-1] = -100
    EpisodeReward = []
    FallCount = 0               # 累計掉落次數

    Gamma = 0.99
    Epsilon = 0.1
    Steps = 1000
    Alpha = 0.05

    if method == 'qlearn':
        for episode in range(episodes):
            print('now_episode = ', episode, end="")
            ActionValue, Epi_Reward, fall = qlearn(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            EpisodeReward.append(Epi_Reward)
            FallCount += fall
            # 即時更新畫面上的掉落次數
            info_label.config(text=f'掉入懸崖次數：{FallCount} / {episode+1} 回合')
            root.update()
    elif method == 'sarsa':
        for episode in range(episodes):
            print('now_episode = ', episode, end="")
            ActionValue, Epi_Reward, fall = sarsa(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            EpisodeReward.append(Epi_Reward)
            FallCount += fall
            # 即時更新畫面上的掉落次數
            info_label.config(text=f'掉入懸崖次數：{FallCount} / {episode+1} 回合')
            root.update()
    else:
        print('No this method.')
        return

    EpisodeReward = np.array(EpisodeReward)
    print('\n掉入懸崖總次數 =', FallCount, '/', episodes)
    print(ActionValue)
    return ActionValue

def draw_map():
    mycanvas.delete("all")
    mycanvas.create_rectangle(10, 10, 1210, 410)

    for i in range(110, 410, 100):
        mycanvas.create_line(10, i, 1210, i)

    for i in range(110, 1210, 100):
        mycanvas.create_line(i, 10, i, 410)

    mycanvas.create_rectangle(10, 310, 110, 410, fill='#ccffcd')
    mycanvas.create_rectangle(1110, 310, 1210, 410, fill='#ccf')

def drawline(reward):
    for i in range(4):
        for j in range(12):
            mycanvas.delete(str(j+(12*i))+'_0')
            mycanvas.delete(str(j+(12*i))+'_1')
            mycanvas.delete(str(j+(12*i))+'_2')
            mycanvas.delete(str(j+(12*i))+'_3')

    now_state = 36
    i, j = 60, 360
    while (now_state != 47):
        m = max(reward[now_state])
        if m == reward[now_state][0]:
            mycanvas.create_line(i, j, i, j-100, fill='#fcc', width=5)
            j = j-100
            now_state = now_state-12
        elif m == reward[now_state][1]:
            mycanvas.create_line(i, j, i-100, j, fill='#fcc', width=5)
            i = i-100
            now_state = now_state-1
        elif m == reward[now_state][2]:
            mycanvas.create_line(i, j, i, j+100, fill='#fcc', width=5)
            j = j+100
            now_state = now_state+12
        elif m == reward[now_state][3]:
            mycanvas.create_line(i, j, i+100, j, fill='#fcc', width=5)
            i = i+100
            now_state = now_state+1

    for i in range(4):
        for j in range(12):
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(reward[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(reward[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(reward[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(reward[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')

    root.update()


def Q_learn():
    draw_map()
    info_label.config(text='掉入懸崖次數：0')      # 重置顯示
    ActionValue = np.zeros([48, 4])

    for i in range(4):
        for j in range(12):
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')

    q_reward = main(1000, 'qlearn')
    drawline(q_reward)

def SARSA():
    draw_map()
    info_label.config(text='掉入懸崖次數：0')      # 重置顯示
    ActionValue = np.zeros([48, 4])

    for i in range(4):
        for j in range(12):
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')

    s_reward = main(1000, 'sarsa')
    drawline(s_reward)


root = tk.Tk()
root.title('Cliff_Walking')
mycanvas = tk.Canvas(root, width=1220, height=420)
mycanvas.pack()

draw_map()

button = tk.Button(root, text='Q_learn', command=Q_learn)
button.pack()

button = tk.Button(root, text='SARSA', command=SARSA)
button.pack()

# 新增：顯示掉落次數的標籤
info_label = tk.Label(root, text='掉入懸崖次數：0', font=('Arial', 14), fg='red')
info_label.pack(pady=5)

root.mainloop()
