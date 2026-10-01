import tkinter as tk  # 匯入 tkinter 圖形介面套件並取別名 tk，用來建立視窗、畫布與按鈕
import numpy as np  # 匯入 numpy 數值運算套件並取別名 np，用來建立與更新 Q 表(48x4 的二維陣列)

#import numpy as np  # 再次匯入 numpy，與上面那行完全重複，屬於多餘的程式碼，可以刪掉
#import os, time  # 匯入 os 與 time 模組；本程式實際上完全沒有用到 os
#import random  # 匯入 Python 內建的 random 亂數模組；本程式實際用的是 np.random，所以這行也沒用到
import time  # 又一次匯入 time 模組(與第 5 行重複)，只有在下面被註解掉的 time.sleep() 才會用到

## defint function
def TransMat(now_state, action):  # 定義狀態轉移函式：輸入目前狀態編號與動作，回傳執行動作後的下一個狀態編號
    max_row = 4  # 懸崖地圖的總列數，共 4 列
    max_col = 12  # 懸崖地圖的總行數，共 12 行；因此總共有 4*12=48 個狀態，編號 0~47
    now_row = int(now_state/max_col)  # 由狀態編號換算出目前所在的「列」索引(0~3)，用整數除法：狀態 36 → 36/12 = 3
    now_col = (now_state%max_col)  # 由狀態編號換算出目前所在的「行」索引(0~11)，用取餘數：狀態 36 → 36%12 = 0

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:  # 邊界檢查：若換算出的行列超出地圖範圍就視為錯誤(嚴格來說應該用 <= 比較才正確，這裡的條件其實有點寬鬆)
        print('index error')  # 印出索引錯誤訊息，提示傳進來的狀態編號不合法
        return  # 直接返回 None，中止這次的狀態轉移計算

    col = now_col  # 先把「下一個位置的行」初始化為目前的行，若動作不合法(撞牆)就會維持原地
    row = now_row  # 先把「下一個位置的列」初始化為目前的列，若動作不合法(撞牆)就會維持原地
    if action == 0 and now_row > 0:  # 動作 0 代表「上」，且目前不在最上面那一列(還有空間往上走)才移動
        row -= 1 #上  # 列索引減 1，代表往上移動一格(畫面上方是 row 較小的方向)
    elif action == 1 and now_col > 0:  # 動作 1 代表「左」，且目前不在最左邊那一行才移動
        col -= 1 #左  # 行索引減 1，代表往左移動一格
    elif action == 2 and (max_row-1) > now_row:  # 動作 2 代表「下」，且目前的列小於最後一列(3)才移動，避免走出地圖底部
        row += 1 #下  # 列索引加 1，代表往下移動一格
    elif action == 3 and (max_col-1) > now_col:  # 動作 3 代表「右」，且目前的行小於最後一行(11)才移動，避免走出地圖右側
        col += 1 #右  # 行索引加 1，代表往右移動一格
    next_state = row*max_col + col  # 把新的(列, 行)重新換算回單一的狀態編號：狀態 = 列*12 + 行
    return next_state  # 回傳下一個狀態的編號給呼叫者

def qlearn(action_value, reward, steps, gamma, alpha, epsilon):  # 定義 Q-learning 演算法：跑完一個回合並回傳更新後的 Q 表與該回合累積報酬
    # initialize setting
    record = []  # 建立空清單，用來記錄這個回合中每一步的 [狀態, 動作, 獎勵, 下一狀態]
    state = 36  # 設定起始狀態為 36，也就是地圖左下角的起點(綠色格子)
    for step in range(steps):  # 迴圈執行最多 steps 步(本程式為 1000 步)，超過就強制結束這個回合
        # get next information
        action = GetAction(action_value, epsilon, state)  # 依 ε-greedy 策略，根據目前 Q 表在 state 選出一個要執行的動作
        next_state = TransMat(state, action)  # 呼叫狀態轉移函式，算出執行該動作後會走到哪個狀態
        record.append([state, action, reward[next_state], next_state])  # 把這一步的經驗存進 record：目前狀態、動作、抵達下一狀態所得的獎勵、下一狀態
        # update action value
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step], alpha, gamma)  # 用 Q-learning 更新公式計算新的 Q 值，並寫回 Q 表對應的 [狀態, 動作] 位置

        #print("action_value=[",state," , ",action,"]=",action_value[state, action])  # 被註解掉的除錯用程式碼：原本會印出剛剛更新的那個 Q 值

        a2 = np.around(action_value,decimals=2)  # 把整張 Q 表四捨五入到小數點後 2 位，做出一份專供畫面顯示用的副本(避免畫面上出現超長浮點數)

        if action == 0:  # 若這一步執行的是動作 0(上)，就更新該格子「上方」位置顯示的 Q 值
            mycanvas.delete(str(state)+'_0')  # 先用標籤(tag)刪除畫布上該格上方原本的舊數字，避免新舊文字重疊
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25,fill='#f00',font=('Arial', 20, 'bold'),text=a2[state, action],tag=str(state)+'_0')  # 在該格上方以紅色粗體 20 級字畫出新的 Q 值，做出「剛剛更新」的閃爍強調效果；座標由狀態編號換算(每格 100 像素、整體偏移 10)
            root.update()  # 強制 tkinter 立即重繪畫面，讓紅色數字馬上顯示出來(否則要等主迴圈空閒才會更新)
            #time.sleep(0.01)  # 被註解掉的暫停 0.01 秒：取消註解可以放慢動畫速度，方便觀察學習過程
            mycanvas.delete(str(state)+'_0')  # 刪除剛剛畫的紅色強調文字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25,fill='#000',text=a2[state, action],tag=str(state)+'_0')  # 用黑色一般字體重畫同一個 Q 值，讓它回復成平常的顯示樣式
        elif action == 1:  # 若這一步執行的是動作 1(左)，就更新該格子「左方」位置顯示的 Q 值
            mycanvas.delete(str(state)+'_1')  # 刪除該格左方原本的舊數字
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50,fill='#f00',font=('Arial', 20, 'bold'),text=a2[state, action],tag=str(state)+'_1')  # 在格子左側以紅色粗體畫出更新後的 Q 值(x 偏移 25、y 偏移 50，正好在格子左邊中央)
            root.update()  # 立即刷新畫面顯示紅字
            #time.sleep(0.01)  # 被註解掉的延遲，用來放慢動畫
            mycanvas.delete(str(state)+'_1')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50,fill='#000', text=a2[state, action],tag=str(state)+'_1')  # 以黑色一般字重畫該 Q 值
        elif action == 2:  # 若這一步執行的是動作 2(下)，就更新該格子「下方」位置顯示的 Q 值
            mycanvas.delete(str(state)+'_2')  # 刪除該格下方原本的舊數字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75,fill='#f00',font=('Arial', 20, 'bold'), text=a2[state, action],tag=str(state)+'_2')  # 在格子下側以紅色粗體畫出更新後的 Q 值(y 偏移 75，位於格子下方中央)
            root.update()  # 立即刷新畫面顯示紅字
            #time.sleep(0.01)  # 被註解掉的延遲
            mycanvas.delete(str(state)+'_2')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75,fill='#000', text=a2[state, action],tag=str(state)+'_2')  # 以黑色一般字重畫該 Q 值
        elif action == 3:  # 若這一步執行的是動作 3(右)，就更新該格子「右方」位置顯示的 Q 值
            mycanvas.delete(str(state)+'_3')  # 刪除該格右方原本的舊數字
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50,fill='#f00',font=('Arial', 20, 'bold'), text=a2[state, action],tag=str(state)+'_3')  # 在格子右側以紅色粗體畫出更新後的 Q 值(x 偏移 75，位於格子右邊中央)
            root.update()  # 立即刷新畫面顯示紅字
            #time.sleep(0.01)  # 被註解掉的延遲
            mycanvas.delete(str(state)+'_3')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50,fill='#000', text=a2[state, action],tag=str(state)+'_3')  # 以黑色一般字重畫該 Q 值

        # update for next state
        state = next_state  # 把「目前狀態」推進到下一個狀態，準備進行下一步的學習
        if state == 47:  # 判斷是否走到狀態 47，也就是地圖右下角的終點(藍色格子)
            print(",[success]")  # 印出成功訊息，代表這個回合順利抵達目標
            break  # 跳出 for 迴圈，結束這個回合
        elif state > 36:  # 判斷狀態編號是否大於 36，也就是掉進 37~46 的懸崖區(47 已在上一個條件被攔下)
            print(",[fail]")  # 印出失敗訊息，代表這個回合掉下懸崖
            break  # 跳出 for 迴圈提前結束回合(注意：標準做法是把 agent 傳送回起點繼續走，這裡改成直接結束回合)
    # episode reward
    record = np.array(record,dtype=object)#<<<---------任意值  # 把記錄清單轉成 numpy 陣列以便做欄位切片；因為每列元素型別不一致(動作可能是陣列)，所以指定 dtype=object
    epi_reward = np.sum(record[:,2])  # 取出第 2 欄(所有步驟的獎勵)並加總，得到這個回合的累積報酬
    return action_value, epi_reward  # 回傳更新後的 Q 表與這回合的累積報酬

def sarsa(action_value, reward, steps, gamma, alpha, epsilon):  # 定義 SARSA 演算法：與 Q-learning 結構類似，但屬於 on-policy
    # initialize setting
    record = []  # 建立空清單，記錄每一步的 [狀態, 動作, 獎勵, 下一狀態, 下一動作]，比 Q-learning 多了「下一動作」
    state = 36  # 設定起始狀態為 36(左下角起點)
    action = GetAction(action_value, epsilon, state)  # SARSA 的特色：在進入迴圈前就先用 ε-greedy 選好第一個動作
    for step in range(steps):  # 最多執行 steps 步，超過就強制結束本回合
        # get next information
        next_state = TransMat(state, action)  # 執行目前的動作，算出會走到哪個狀態
        next_action = GetAction(action_value, epsilon, next_state)  # 再用 ε-greedy 在新狀態選出「下一個實際要執行的動作」，這正是 SARSA 更新公式所需要的
        record.append([state, action, reward[next_state], next_state, next_action])  # 把 (S, A, R, S', A') 這五元組完整記錄下來，SARSA 名稱即由此而來
        # update action value
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step], alpha, gamma)  # 用 SARSA 更新公式計算新的 Q 值並寫回 Q 表

        a2 = np.around(action_value,decimals=2)  # 產生四捨五入到小數點後 2 位的 Q 表副本，供畫面顯示使用

        if action == 0:  # 若執行的是動作 0(上)，更新該格上方顯示的數值
            mycanvas.delete(str(state)+'_0')  # 刪除該格上方的舊數字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25,fill='#f00',font=('Arial', 20, 'bold'),text=a2[state, action],tag=str(state)+'_0')  # 以紅色粗體畫出更新後的 Q 值做視覺強調
            root.update()  # 立即刷新畫面
            #time.sleep(0.05)  # 被註解掉的延遲 0.05 秒，可放慢動畫
            mycanvas.delete(str(state)+'_0')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+25,fill='#000',text=a2[state, action],tag=str(state)+'_0')  # 用黑色一般字重畫，回復正常樣式
        elif action == 1:  # 若執行的是動作 1(左)，更新該格左方顯示的數值
            mycanvas.delete(str(state)+'_1')  # 刪除該格左方的舊數字
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50,fill='#f00',font=('Arial', 20, 'bold'),text=a2[state, action],tag=str(state)+'_1')  # 以紅色粗體畫出新的 Q 值
            root.update()  # 立即刷新畫面
            #time.sleep(0.05)  # 被註解掉的延遲
            mycanvas.delete(str(state)+'_1')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+25, 10+(state//12*100)+50,fill='#000', text=a2[state, action],tag=str(state)+'_1')  # 用黑色一般字重畫
        elif action == 2:  # 若執行的是動作 2(下)，更新該格下方顯示的數值
            mycanvas.delete(str(state)+'_2')  # 刪除該格下方的舊數字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75,fill='#f00',font=('Arial', 20, 'bold'), text=a2[state, action],tag=str(state)+'_2')  # 以紅色粗體畫出新的 Q 值
            root.update()  # 立即刷新畫面
            #time.sleep(0.05)  # 被註解掉的延遲
            mycanvas.delete(str(state)+'_2')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+50, 10+(state//12*100)+75,fill='#000', text=a2[state, action],tag=str(state)+'_2')  # 用黑色一般字重畫
        elif action == 3:  # 若執行的是動作 3(右)，更新該格右方顯示的數值
            mycanvas.delete(str(state)+'_3')  # 刪除該格右方的舊數字
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50,fill='#f00',font=('Arial', 20, 'bold'), text=a2[state, action],tag=str(state)+'_3')  # 以紅色粗體畫出新的 Q 值
            root.update()  # 立即刷新畫面
            #time.sleep(0.05)  # 被註解掉的延遲
            mycanvas.delete(str(state)+'_3')  # 刪除紅色強調文字
            mycanvas.create_text(10+(state%12*100)+75, 10+(state//12*100)+50,fill='#000', text=a2[state, action],tag=str(state)+'_3')  # 用黑色一般字重畫

        # update for next state
        state = next_state  # 把目前狀態推進到下一個狀態
        action = next_action  # 把目前動作換成剛才選定的下一個動作(SARSA 必須實際執行它，才符合 on-policy 的定義)
        state = next_state  # 重複執行了一次相同的指派(與上面兩行前重複)，屬於多餘的程式碼，不影響結果
        if state == 47:  # 判斷是否抵達終點狀態 47
            print(",[success]")  # 印出成功訊息
            break  # 結束本回合
        elif state > 36:  # 判斷是否掉入 37~46 的懸崖區
            print(",[fail]")  # 印出失敗訊息
            break  # 結束本回合
    # episode reward
    record = np.array(record,dtype=object)#<<<---------任意值  # 將記錄轉成 object 型別的 numpy 陣列，方便用切片取出獎勵欄位
    epi_reward = np.sum(record[:,2])  # 加總第 2 欄的所有獎勵，得到本回合累積報酬
    return action_value, epi_reward  # 回傳更新後的 Q 表與本回合累積報酬

def GetAction(action_value, epsilon, next_state):  # 定義 ε-greedy 動作選擇函式：以機率取捨「利用」與「探索」
    if np.random.rand(1) >= epsilon:  # 產生一個 0~1 的亂數，若大於等於 epsilon(機率 1-ε = 90%)就走「利用」路線
        policy = np.argmax(action_value, axis = 1)  # 對 Q 表每一列(每個狀態)取最大值的索引，得到目前的貪婪策略(每個狀態的最佳動作)
        action = policy[next_state]  # 從策略中取出該狀態對應的最佳動作編號
    else:  # 否則(機率 ε = 10%)走「探索」路線，隨機亂走以發現新的可能性
        action = np.random.randint(0,4)  # 隨機產生 0~3 的整數動作；改成不指定 size，回傳純量，型別才會跟上面 argmax 分支一致(否則 SARSA 把陣列型態的 action 存回 Q 表純量位置時會丟出 ValueError 而中斷訓練)
    return action  # 回傳選定的動作

def ValueUpdate(method, action_value, record, alpha, gamma):  # 定義 Q 值更新函式：依指定方法(qlearn 或 sarsa)計算新的動作價值
    state = record[0]  # 從這一步的記錄中取出「目前狀態」
    action = record[1]  # 取出「執行的動作」
    reward = record[2]  # 取出「獲得的立即獎勵」
    next_state = record[3]  # 取出「轉移到的下一狀態」
    now_value = action_value[state, action]  # 從 Q 表讀出目前這組 (狀態, 動作) 的舊估計值 Q(s,a)
    if method == 'qlearn':  # 若使用 Q-learning(off-policy)
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)  # 計算增量：學習率 α ×(獎勵 + 折扣 γ × 下一狀態的「最大」Q 值 − 舊值)，即用最佳動作來估計未來報酬
    elif method == 'sarsa':  # 若使用 SARSA(on-policy)
        next_action = record[4]  # 取出「下一步實際會執行的動作」
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)  # 計算增量：改用實際會執行的動作 A' 的 Q 值來估計未來報酬，因此會把探索的風險也學進去(策略較保守)
    else:  # 若傳入的方法名稱不在支援清單內
        print('No this method.')  # 印出錯誤訊息提示方法名稱有誤
        return  # 回傳 None 並中止函式
    value = now_value + update_value  # 新的 Q 值 = 舊值 + 增量(時間差分 TD 更新)
    return value  # 回傳更新後的 Q 值

# main
def main(episodes, method):  # 定義主訓練流程：設定環境與超參數，並重複執行指定回合數的學習
    # environment setting
    ActionValue = np.zeros([48, 4])  # 建立 48 個狀態 × 4 個動作的 Q 表，全部初始化為 0
    Reward = np.full(48, -1)  # 建立長度 48 的獎勵陣列，預設每一格都是 -1(每走一步就扣 1 分，鼓勵走最短路徑)
    Reward[37:-1] = -100  # 把索引 37 到 46(不含最後的 47)設為 -100，代表下方那一整排懸崖，掉下去會被重罰
    EpisodeReward = []  # 建立空list，用來收集每個回合的累積報酬，可用於後續觀察學習曲線
    # parameters setting
    Gamma = 0.99  # 折扣因子 γ：越接近 1 代表越重視長期報酬
    Epsilon = 0.1  # 探索率 ε：有 10% 的機率隨機選動作，其餘 90% 選目前認為最好的動作
    Steps = 1000  # 每個回合最多允許走 1000 步，避免 agent 在地圖中無限打轉
    Alpha = 0.05  # 學習率 α：每次更新只朝新估計值移動 5%，數值小則學得慢但較穩定

    # Execute
    if method == 'qlearn':  # 若指定使用 Q-learning 演算法
        for episode in range(episodes):  # 依指定的回合數重複訓練
            print('now_episode = ', episode,end="")  # 印出目前回合編號；end="" 表示不換行，好讓後面的 [success]/[fail] 接在同一行
            ActionValue, Epi_Reward = qlearn(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)  # 執行一個回合的 Q-learning，並把更新後的 Q 表接回來供下一回合繼續使用
            EpisodeReward.append(Epi_Reward)  # 把這回合的累積報酬存入清單
    elif method == 'sarsa':  # 若指定使用 SARSA 演算法
        for episode in range(episodes):  # 依指定的回合數重複訓練
            print('now_episode = ', episode,end="")  # 印出目前回合編號，不換行
            ActionValue, Epi_Reward = sarsa(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)  # 執行一個回合的 SARSA 並取回更新後的 Q 表
            EpisodeReward.append(Epi_Reward)  # 把這回合的累積報酬存入清單
    else:  # 若方法名稱不合法
        print('No this method.')  # 印出錯誤訊息
        return  # 中止函式
    EpisodeReward = np.array(EpisodeReward)  # 把回合報酬清單轉成 numpy 陣列(方便日後畫學習曲線，本程式尚未使用)

    print(ActionValue)  # 在終端機印出訓練完成後的完整 Q 表，方便檢視數值

    return ActionValue  # 回傳訓練好的 Q 表，供外層畫出最佳路徑

def draw_map():  # 定義畫地圖函式：在畫布上繪製 4×12 的格線地圖與起點、終點
    mycanvas.delete("all")  # 清空畫布上所有既有圖形，重畫前先歸零
    mycanvas.create_rectangle(10, 10, 1210, 410)  # 畫出整張地圖的外框矩形，左上角(10,10)、右下角(1210,410)，即 1200×400 像素

    for i in range(110,410,100):  # 從 y=110 開始每隔 100 像素取一次值(110、210、310)，用來畫水平分隔線
        mycanvas.create_line(10, i, 1210, i)  # 畫一條橫貫地圖的水平線，把地圖分成 4 列

    for i in range(110,1210,100):  # 從 x=110 開始每隔 100 像素取值(110~1110)，用來畫垂直分隔線
        mycanvas.create_line(i, 10, i, 410)  # 畫一條貫穿地圖的垂直線，把地圖分成 12 行

    mycanvas.create_rectangle(10, 310, 110, 410, fill='#ccffcd')  # 用淡綠色填滿左下角格子，標示起點(狀態 36)
    mycanvas.create_rectangle(1110, 310, 1210, 410, fill='#ccf')  # 用淡藍色填滿右下角格子，標示終點(狀態 47)

def drawline(reward):  # 定義畫最佳路徑函式：依訓練完成的 Q 表，從起點沿著貪婪策略畫出粉紅色路線並重新標上所有 Q 值

    for i in range(4):  # 逐列走訪地圖的 4 列
        for j in range(12):  # 逐行走訪地圖的 12 行
            mycanvas.delete(str(j+(12*i))+'_0')#上  # 刪除該格上方的舊數值文字
            mycanvas.delete(str(j+(12*i))+'_1')#左  # 刪除該格左方的舊數值文字
            mycanvas.delete(str(j+(12*i))+'_2')#下  # 刪除該格下方的舊數值文字
            mycanvas.delete(str(j+(12*i))+'_3')#右  # 刪除該格右方的舊數值文字，清空後才不會被路線遮住或重疊

    now_state = 36  # 從起點狀態 36 開始追蹤最佳路徑
    i, j = 60 ,360  # 設定畫線的起始座標，(60, 360) 正好是左下角起點格子的中心點
    while (now_state != 47):  # 只要還沒走到終點 47 就持續追蹤(若學到的策略有迴圈，這裡會陷入無窮迴圈導致畫面卡住)
        m = max(reward[now_state])  # 取出目前狀態四個動作中最大的 Q 值，代表貪婪策略會選的那個動作的價值
        if m == reward[now_state][0]:#上  # 若最大值來自動作 0，表示最佳動作是往上
            mycanvas.create_line(i, j, i, j-100, fill='#fcc',width=5)  # 從目前中心點往上畫一條 100 像素長、寬度 5 的粉紅色線段
            j = j-100  # 更新畫線的 y 座標，往上移一格
            now_state=now_state-12  # 更新狀態編號，往上一列相當於減 12
        elif m == reward[now_state][1]:#左  # 若最大值來自動作 1，表示最佳動作是往左
            mycanvas.create_line(i, j, i-100, j, fill='#fcc',width=5)  # 往左畫一條粉紅色線段
            i = i-100  # 更新畫線的 x 座標，往左移一格
            now_state=now_state-1  # 更新狀態編號，往左一行相當於減 1
        elif m == reward[now_state][2]:#下  # 若最大值來自動作 2，表示最佳動作是往下
            mycanvas.create_line(i, j, i, j+100, fill='#fcc',width=5)  # 往下畫一條粉紅色線段
            j = j+100  # 更新畫線的 y 座標，往下移一格
            now_state=now_state+12  # 更新狀態編號，往下一列相當於加 12
        elif m == reward[now_state][3]:#右  # 若最大值來自動作 3，表示最佳動作是往右
            mycanvas.create_line(i, j, i+100, j, fill='#fcc',width=5)  # 往右畫一條粉紅色線段
            i = i+100  # 更新畫線的 x 座標，往右移一格
            now_state=now_state+1  # 更新狀態編號，往右一行相當於加 1

    for i in range(4):  # 路徑畫完後，再逐列走訪 4 列
        for j in range(12):  # 逐行走訪 12 行，把每一格四個方向的最終 Q 值重新標示上去
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(reward[j+(12*i)][0],'.2f'),tag=str(j+(12*i))+'_0')#上  # 在格子上方標出「往上」動作的 Q 值，格式化為小數點後 2 位
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(reward[j+(12*i)][1],'.2f'),tag=str(j+(12*i))+'_1')#左  # 在格子左方標出「往左」動作的 Q 值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(reward[j+(12*i)][2],'.2f'),tag=str(j+(12*i))+'_2')#下  # 在格子下方標出「往下」動作的 Q 值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(reward[j+(12*i)][3],'.2f'),tag=str(j+(12*i))+'_3')#右  # 在格子右方標出「往右」動作的 Q 值

    root.update()  # 強制刷新視窗，讓路徑與所有數值立刻呈現在畫面上


def Q_learn():  # 定義「Q_learn」按鈕的回呼函式：按下按鈕後會重畫地圖並開始 Q-learning 訓練
    draw_map()  # 先把地圖重新畫一次，清除上一次執行留下的線條與數字
    ActionValue = np.zeros([48, 4])  # 建立一張全為 0 的 Q 表，僅用於畫出初始畫面上的 0.00 數值(真正訓練用的 Q 表在 main 裡另外建立)

    for i in range(4):  # 逐列走訪 4 列
        for j in range(12):  # 逐行走訪 12 行，在每格四個方向標上初始值 0.00
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'),tag=str(j+(12*i))+'_0')#上  # 在格子上方畫出「上」的初始 Q 值並掛上識別標籤，方便之後刪除或更新
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'),tag=str(j+(12*i))+'_1')#左  # 在格子左方畫出「左」的初始 Q 值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'),tag=str(j+(12*i))+'_2')#下  # 在格子下方畫出「下」的初始 Q 值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'),tag=str(j+(12*i))+'_3')#右  # 在格子右方畫出「右」的初始 Q 值

    q_reward = main(1000, 'qlearn')  # 呼叫主流程，以 Q-learning 訓練 1000 個回合，並取回訓練完成的 Q 表
    drawline(q_reward)  # 依訓練結果畫出貪婪最佳路徑；Q-learning 通常會學到緊貼懸崖邊緣的最短路徑

def SARSA():  # 定義「SARSA」按鈕的回呼函式：按下按鈕後重畫地圖並開始 SARSA 訓練
    draw_map()  # 重新畫地圖，清掉上一次執行的殘留內容
    ActionValue = np.zeros([48, 4])  # 建立全 0 的 Q 表，僅供畫出畫面上的初始 0.00 數值使用

    for i in range(4):  # 逐列走訪 4 列
        for j in range(12):  # 逐行走訪 12 行
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'),tag=str(j+(12*i))+'_0')#上  # 標出「上」的初始 Q 值
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'),tag=str(j+(12*i))+'_1')#左  # 標出「左」的初始 Q 值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'),tag=str(j+(12*i))+'_2')#下  # 標出「下」的初始 Q 值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'),tag=str(j+(12*i))+'_3')#右  # 標出「右」的初始 Q 值

    s_reward = main(1000, 'sarsa')  # 呼叫主流程，以 SARSA 訓練 1000 個回合並取回 Q 表
    drawline(s_reward)  # 畫出 SARSA 學到的路徑；SARSA 因為把探索風險納入考量，通常會選擇離懸崖較遠的保守路線


root = tk.Tk()  # 建立 tkinter 主視窗物件，這是整個 GUI 的根容器
root.title('Cliff_Walking')  # 設定視窗標題列顯示的文字為 Cliff_Walking
mycanvas = tk.Canvas(root, width=1220, height=420)  # 在主視窗中建立一塊 1220×420 像素的畫布，用來繪製地圖、數值與路徑
mycanvas.pack()  # 用 pack 版面管理器把畫布放入視窗中並實際顯示出來

draw_map()  # 程式一啟動就先畫出空白的地圖，讓使用者看到初始的格線地圖

button = tk.Button(root, text='Q_learn', command=Q_learn)  # 建立一個顯示「Q_learn」的按鈕，按下時會呼叫 Q_learn 函式開始訓練
button.pack()  # 把這個按鈕放進視窗排版中顯示出來

button = tk.Button(root, text='SARSA', command=SARSA)  # 建立顯示「SARSA」的按鈕，按下時呼叫 SARSA 函式；注意這裡沿用同一個變數名 button，會覆蓋上一個參考，但按鈕本身已經 pack 進視窗所以仍正常運作
button.pack()  # 把 SARSA 按鈕放進視窗排版中顯示出來

root.mainloop()  # 啟動 tkinter 事件主迴圈，程式會停在這裡持續等待使用者操作，直到視窗被關閉為止
